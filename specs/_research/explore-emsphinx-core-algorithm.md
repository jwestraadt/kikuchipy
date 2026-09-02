# EMSphInx core spherical-indexing algorithm — port reference

Source repo: `c:/Users/westraadt.1/Repos/EMSphInx` (branch `master`, HEAD `60f3517`).
All line numbers below refer to those files. Everything is header-only C++ templates on `Real` (the EBSD driver instantiates `Real = double`, see `programs/index_ebsd.cpp:52`).

Global conventions (`include/constants.hpp:52-67`):
- Orientations are **passive rotations sample → crystal**.
- `pijk = +1`.
- ZYZ Euler `eu = {alpha, beta, gamma}` = rotate frame by alpha about z, then beta about y', then gamma about z''.
- `Constants<Real>::pi`, `pi2 = 2*pi`, `pi_2 = pi/2`.

---

## 1. Square grids on the sphere and the DiscreteSHT

File: `include/sht/square_sht.hpp` (1166 lines).

### 1.1 Grid requirements (`:44-52`)
A "square" spherical grid must satisfy:
- rings of constant latitude map to square rings;
- rings are symmetric across the equator;
- points within a ring are equally spaced in azimuth;
- **side length `dim` is odd** (enforced: `:339` `if(dim % 2 == 0) throw std::domain_error("only odd side lengths are supported")`; also `dim < 3` throws, `:338`);
- ring point counts from pole → equator are `1, 8, 16, 24, 32, ...` i.e. `N_phi(y) = max(1, 8*y)`;
- the first point of each ring is at azimuth `phi = 0`;
- the outer boundary of the square is the equator (shared by both hemispheres → "double cover").

Storage: one hemisphere is a `dim x dim` **row-major** array; index `= j*dim + i`, with `X = i/(dim-1)`, `Y = j/(dim-1)` both in `[0,1]`. A full sphere is `2*dim*dim` values: north hemisphere first, then south (`analyze(pts, ...)` at `:136` splits at `pts + dim*dim`).

Ring count `Nt = (dim+1)/2`; ring `0` = north pole (1 point), ring `Nt-1 = (dim-1)/2` = equator (`4*(dim-1) = 8*(Nt-1)` points).

`ringNum(dim, idx)` (`:1144-1160`): Chebyshev distance from the centre pixel,
```
y = idx/dim; x = idx - y*dim; d2 = dim/2
ring = max(|x-d2|, |y-d2|)
```

### 1.2 Two layouts (`:58-61`)
```
enum class Layout { Lambert,   // equal-area (Rosca) square Lambert
                    Legendre };// iso-latitude rings at Legendre roots (+ poles)
```
- **Lambert**: all pixels have equal solid angle. `DiscreteSHT::Lambert(dim)` uses `mBw = (dim-1)/2` (`:112`).
- **Legendre**: the ring **z-coordinates are the roots of `P_{dim-2}(z)`** plus an explicit pole. Because Gauss–Legendre quadrature is exact, you get roughly *double* the bandwidth for the same `dim`. `DiscreteSHT::Legendre(dim)` uses `mBw = dim - 2` (`:107`; "poles aren't used"). **The indexer uses Legendre exclusively.**

### 1.3 Square-Lambert (Rosca) mapping (`:584-736`)

`squareToSphere(X, Y, x, y, z)` (`:614-642`), `X,Y in [0,1]`:
```
sX = 2X-1 ; sY = 2Y-1 ; aX=|sX| ; aY=|sY| ; vMax = max(aX,aY)
if vMax <= eps:            (x,y,z) = (0,0,1)
elif aX <= aY:  q = sY*sqrt(2 - sY^2) ; qq = (pi/4)*sX/sY
                x = q*sin(qq) ;  y = q*cos(qq)
else:           q = sX*sqrt(2 - sX^2) ; qq = (pi/4)*sY/sX
                x = q*cos(qq) ;  y = q*sin(qq)
z = 1 - vMax^2
normalize (x,y,z)          # analytically already unit; normalization only kills round-off
```
Throws if `vMax > 1 + eps`.

`sphereToSquare(x, y, z, X, Y)` (`:591-606`) — inverse, uses `fZ = |z|` (so it maps *both* hemispheres onto the same square):
```
if |z| == 1: X = Y = 0.5
elif |y| <= |x|:  X = copysign(sqrt(1-|z|), x)*0.5 ; Y = X*atan(y/x)/(pi/4) + 0.5 ; X += 0.5
else:             Y = copysign(sqrt(1-|z|), y)*0.5 ; X = Y*atan(x/y)/(pi/4) + 0.5 ; Y += 0.5
```

`lambert::cosLats(dim, lat)` (`:648-660`) — closed form for odd `dim`:
```
cos(theta_y) = 1 - (2y/(dim-1))^2 ,   y = 0 .. (dim-1)/2
```
(the code accumulates `numer -= delta; delta += 8` starting from `numer=(dim-1)^2, delta=4`, which is exactly `1 - 4y^2/(dim-1)^2`).

`lambert::normals(dim, xyz)` (`:665-675`): `xyz[3*(j*dim+i)+{0,1,2}] = squareToSphere(i/(dim-1), j/(dim-1))`.

`lambert::solidAngles(dim, omg)` (`:681-736`): exact per-pixel solid angle of the spherical quad spanned by the 4 pixel corners (`+/- 0.5/(dim-1)` in square space, clipped at the equator), via Mazonka's formula (product of complex numbers `(b_j c_j - a_j, d_j)`, `omega = arg(product)`); result is normalized by the average pixel solid angle `4*pi/totalPixels`, with `totalPixels = 2*dim^2 - 4*(dim-1)`. Edge pixels get `factor *= 2` per equatorial edge. Uses the 8-fold mirror symmetry of the square.

### 1.4 Square-Legendre grid (`:739-933`)

`legendre::roots(n, lat)` (`:746-818`): the positive half of the roots of `P_n`, computed as eigenvalues of the symmetric tridiagonal Jacobi matrix (diagonal all zero, sub-diagonal `b_i = i/sqrt(4i^2-1)`) by **bisection with a Sturm sequence** (Barth/Martin/Wilkinson). `m1 = n/2` eigenvalues computed largest → smallest; for odd `n` the middle root `x[m1] = 0` is set explicitly. Iteration cap `n*32`, tolerance `eps1 = eps2 = machine eps`, `eps2 = eps1/2 + 7*eps` .

`cosLats(dim, Layout::Legendre)` (`:1069-1081`):
```
cLats[0] = 1                      # explicit pole
legendre::roots(dim-2, &cLats[1]) # (dim-1)/2 values, descending, last == 0 (equator)
```
So ring `y>0` sits at `z = root_y` of `P_{dim-2}`.

`legendre::normals(dim, xyz)` (`:823-869`) — this is the key difference from Lambert: **the azimuthal placement is the square-ring azimuth, but the latitude comes from the Legendre table**:
```
half = dim/2
cosLats = roots(dim-2)                # (dim-1)/2 values
for j,i in [0,dim):
    rj = j-half ; aj=|rj| ; ri = i-half ; ai=|ri| ; ar = max(ai,aj)   # ring number
    if ar == 0: n = (0,0,1)
    else:
        sX = ri/ar ; sY = rj/ar                     # in [-1,1] on the ring perimeter
        if ai <= aj:  qq = (pi/4)*sX*sY ; x = sY*sin(qq) ; y = sY*cos(qq)
        else:         qq = (pi/4)*sY*sX ; x = sX*cos(qq) ; y = sX*sin(qq)
        h = hypot(x,y)
        n[2] = cosLats[ar-1]
        n[0] = n[1] = sqrt(1 - n[2]^2)
        n[0] *= x/h ;  n[1] *= y/h
```
(Only odd `dim` supported, `:825`.) Note the same azimuth distribution as Lambert (equal `pi/4 * s` spacing around the square ring) but re-scaled radii.

`legendre::boundingInds(dim, zLat, n, inds)` (`:876-933`) — returns the 4 grid indices bounding a direction (used by `toLambert`). Uses `lower_bound` with `greater<>` on `zLat` to find the bracketing rings, then walks around the ring by `ceil/floor(theta * ring)` where `theta = 4*atan(min(|nx|,|ny|)/max(|nx|,|ny|))/pi`.

`solidAngles(dim, type)` (`:1105-1138`) — **per-ring** version returning `Nt` values (ring solid angle / average pixel solid angle):
```
cLats  = cosLats(dim, type)                                  # size Nt
cSplits[i] = cos((theta_i + theta_{i+1})/2)
           = (sqrt((1+cA)(1+cB)) - sqrt((1-cA)(1-cB)))/2     # i = 0..Nt-2
cSplits.push_back(-cSplits.back())                           # mirror across equator -> size Nt
c -> (1 - c)                                                 # spherical cap area / 2pi
adjacent_difference(cSplits)                                 # ring band area / 2pi
avgPix = 2/numPix,  numPix = 2*dim^2 - 4*(dim-1)
cSplits[0] /= avgPix*1
num = 8 ; for i>=1: cSplits[i] /= avgPix*num ; num += 8
```
The last entry (equator) is a *full* band straddling the equator, so equatorial pixels' weights must be halved by the caller when only one hemisphere copy is used.

### 1.5 Ring extraction / insertion (`:942-1014`)

`readRing(dim, ring, ptr, buff)` / `writeRing(...)`; for odd `dim` (`even = 0`):
```
side  = 2*ring + 1
pole  = (dim*dim)/2                    # centre pixel index
start = pole + ring                    # phi = 0 point (+x, y = 0)
quad1 = start + dim*ring               # (+x,+y) corner
quad2 = quad1 - (side-1)               # (-x,+y)
quad3 = quad2 - dim*(side-1)           # (-x,-y)
quad4 = quad3 + (side-1)               # (+x,-y)
b1 = ring; b2 = b1+side-1; b3 = b2+side-1; b4 = b3+side-1
```
Buffer index `0` is the `phi = 0` point; the buffer walks **counter-clockwise (increasing phi)**. Length returned = `4*(side-1) = 8*ring` (and `1` for `ring == 0`). Azimuth of buffer slot `p`: `phi_p = 2*pi*p/(8*ring)`.

For a NumPy port the simplest faithful approach is to precompute, once per `dim`, an integer index array `ringIdx[y] -> int[8y]` by running this same logic.

### 1.6 Quadrature weights `computeWeightsSkip` (`:1022-1063`)

This solves the Sneeuw (1994) linear system for iso-latitude quadrature weights:
```
nMat = (dim+1)/2 - 1 = Nt - 1
theta_list = ring latitudes with ring `skp` removed  (indices <skp use lat[i], >=skp use lat[i+1])
A[j][i] = cos(2*j*theta_i)              # Chebyshev recursion T_n(x), x_i = 2*lat_i^2 - 1
b[0] = 1 ; b[j] = -1/(4*j^2 - 1)  j>=1
solve::lu(A, w, b, nMat)                # w sums to 1 (north hemisphere half of 2)
assert(sum(w) - 1 <= cbrt(eps)/64)      # else throw "insufficient precision..."
# re-insert the skipped ring as a zero weight
w = [w[0..skp), 0, w[skp..nMat)]        # length Nt
gridPoints = 2*dim^2 - 4*(dim-1)
wn = 2*pi*4/gridPoints                  # == 8*pi/gridPoints
w0 = wn * ((dim-2)*dim + 2)             # == 4*pi exactly for odd dim
w[0]  *= w0/1
w[i]  *= w0/(8*i)   for i >= 1
```
**Net result (odd `dim`):** `w_y = 4*pi * what_y / N_phi(y)` with `N_phi(y) = max(1, 8y)` and `sum_y what_y = 1`. The `1/N_phi` factor is exactly the FFTW forward-transform normalisation, so `synthesize` needs no extra scaling.

Why "skip": ring `y` has `8y` real samples, so the real FFT has a Nyquist bin at `m = 4y` whose imaginary part is structurally zero — that ring is "problematic" for order `m = 4y`. The code therefore precomputes `Nw = (dim-2)/4 + 1` weight sets, one per skipped ring, and in `analyze`/`synthesize` uses `wy[(m/4)*Nt + y]` (`:439`). For the Legendre layout only `skip = 0` (skip the pole) is computed and copied to all `Nw` slots (`:376-378`) since Gauss–Legendre weights are exact.

`Constants` layout (`:310-328`):
```
dim, Nt=(dim+1)/2, maxL, Nw=(dim-2)/4+1
wy   : Real[Nt*Nw]
cosTy: Real[Nt]
amn, bmn : Real[maxL*maxL]
ffts : RealFFT plans of length max(1, 8y), one per ring y
```
Bandwidth validation (`:340-345`):
```
Legendre:  limit = 2*(Nt-1) + (1 - dim%2)  ->  maxL < dim-1   (i.e. maxL <= dim-2)
Lambert :  maxL < Nt = (dim+1)/2
```

### 1.7 `amn`/`bmn` — Schaeffer (2013) normalized ALF recursion (`:347-373`)
```
k4p  = 1/(4*pi)
kamm = 1                                 # prod_{k=1}^{m} (2k+1)/(2k) for m=0
for m in [0, maxL):
    amn[m][m] = sqrt(kamm * k4p)                       # eq 16
    kamm *= (2m+3)/(2m+2)
    if m+1 == maxL: break
    amn[m][m+1] = sqrt((4*(m+1)^2 - 1)/((m+1)^2 - m^2))# eq 17
    for n in [m+2, maxL):
        amn[m][n] = sqrt((4n^2 - 1)/(n^2 - m^2))                       # eq 17
        bmn[m][n] = sqrt(((2n+1)/(2n-3)) * (((n-1)^2 - m^2)/(n^2 - m^2)))# eq 18
```
On-the-fly evaluation at `x = cos(theta_y)`, `r1x2 = sqrt(1-x^2)`:
```
P^m_m       = amn[m][m] * (1-x^2)^(m/2)     # tracked via kpmm *= r1x2 each m   (eq 13)
P^m_{m+1}   = amn[m][m+1] * x * P^m_m                                          # (eq 14)
P^m_n       = amn[m][n] * x * P^m_{n-1} - bmn[m][n] * P^m_{n-2}                # (eq 15)
```
These are **fully normalized** `Ybar^m_n` radial parts **without the Condon–Shortley `(-1)^m`** — which is why odd `m` are negated when applying the ring weight (`:439`, `:554-555`).

### 1.8 `analyze` — forward SHT (`:414-486`)

Output layout: `alm[stM*m + l]`, i.e. **m-major, l-minor**, `stM` defaults to `bw`. Only `m >= 0` stored; `a^l_{-m} = conj(a^l_m) * (-1)^m` (`:120`, `:413`).

```
maxL = bw or shtLut->maxL ;  stride = stM or maxL
alm[:] = 0
for y in [0, Nt):
    Npy  = max(1, 8y) ; fftN = Npy/2 + 1
    rN = readRing(dim, y, nh) ; rS = readRing(dim, y, sh)
    cN = rfft(rN) ; cS = rfft(rS)                    # UNNORMALIZED (FFTW r2c)
    mLim = min(maxL, fftN)
    for m in [0, mLim):
        w = wy[(m//4)*Nt + y] * (-1)^m
        nPt = cN[m]*w ; sPt = cS[m]*w
        cN[m] = (nPt + sPt)*0.5     # G^+  -> even (l+m)  (harmonics symmetric  across equator)
        cS[m] = (nPt - sPt)*0.5     # G^-  -> odd  (l+m)  (harmonics antisym.  across equator)
    x = cosTy[y] ; r1x2 = sqrt(1-x^2) ; kpmm = 1
    for m in [0, mLim):
        gS = cN[m] ; gA = cS[m]
        p2 = amn[m][m]*kpmm ; kpmm *= r1x2
        alm[m*stride + m] += gS * p2
        if m+1 == maxL: break
        p1 = amn[m][m+1]*x*p2
        alm[m*stride + m+1] += gA * p1
        for n in [m+2, maxL):
            p = amn[m][n]*x*p1 - bmn[m][n]*p2 ; p2 = p1 ; p1 = p
            alm[m*stride + n] += ((n+m)%2==0 ? gS : gA) * p
```
Sanity value: for `f == 1`, `a^0_0 = sqrt(4*pi)`.

### 1.9 `synthesize` — inverse SHT (`:495-572`)
```
for y in [0, Nt):
    Npy = max(1,8y) ; fftN = Npy/2 + 1 ; mLim = min(maxL, fftN)
    x = cosTy[y] ; r1x2 = sqrt(1-x^2) ; kpmm = 1
    for m in [0, mLim):
        fS = fA = 0
        p2 = amn[m][m]*kpmm ; kpmm *= r1x2 ; fS += alm[m*stride+m]*p2
        if m+1 == maxL: break
        p1 = amn[m][m+1]*x*p2 ; fA += alm[m*stride+m+1]*p1
        for n in [m+2, maxL): recursion; (n+m)%2==0 ? fS : fA  += alm[m*stride+n]*p
        cWrk1[m] = fS ; cWrk2[m] = fA
    for m in [0, mLim):
        sigma = cWrk1[m]*(-1)^m ; delta = cWrk2[m]*(-1)^m
        cWrk1[m] = sigma + delta      # north
        cWrk2[m] = sigma - delta      # south
    if fftN >= maxL: zero cWrk1[maxL .. fftN], cWrk2[maxL .. fftN]
    writeRing(dim, y, nh, irfft(cWrk1))    # UNNORMALIZED FFTW c2r
    writeRing(dim, y, sh, irfft(cWrk2))
```

**NumPy note:** `np.fft.rfft` matches FFTW `r2c`; `np.fft.irfft(z, n)` differs from FFTW `c2r` by a factor `1/n` — multiply by `n` (or equivalently absorb it, but do NOT double-apply the `1/N_phi` already in `w_y`).

---

## 2. Wigner d / D functions — `include/sht/wigner.hpp`

### 2.1 Conventions (`:43-63`)
```
d^j_{k,m}(beta) = sqrt(((j+k)!(j-k)!)/((j+m)!(j-m)!)) * cos(beta/2)^(k+m) * sin(beta/2)^(k-m)
                  * P^{k-m, k+m}_{j-k}(cos beta)          [Jacobi polynomial]  (Fukushima eq 1)
D^j_{k,m}(alpha,beta,gamma) = d^j_{k,m}(beta) * exp(I*m*alpha) * exp(I*k*gamma)
```
Equivalent to Mathematica `WignerD[{j,k,m}, gamma, beta, alpha]` (`:100`). Rotation of harmonics: `aRot^l_k = sum_m a^l_m D^l_{k,m}(ZYZ)` (`:52`).
Notation uses `j` (degree), `k`, `m` (orders) — not `l`.

### 2.2 Recursion coefficients (Fukushima 2016; `:204-286`)
```
u_jkm_0(j,k,m,tc) = -tc*((j-1)*j) - (k*m - (j-1)*j)          # 0 <= beta < pi/2 , tc = 1-t
u_jkm_1(j,k,m)    = -k*m                                     # beta == pi/2
u_jkm_2(j,k,m,t)  =  t*((j-1)*j) - k*m                       # beta >  pi/2
v_jkm(j,k,m) = sqrt((j+k-1)(j-k-1)(j+m-1)(j-m-1)) * j
w_jkm(j,k,m) = 1 / ( sqrt((j+k)(j-k)(j+m)(j-m)) * (j-1) )
a_jkm_*      = w_jkm * ( u_jkm_* * (2j-1) )                  # eq 11
b_jkm        = w_jkm * v_jkm                                 # eq 12
u_km_0(k,m,tc) = -tc*(k+1) - (m-1-k)
u_km_1(k,m)    = -m
u_km_2(k,m,t)  =  t*(k+1) - m
a_km_*         = sqrt((2k+1)/((k+m+1)(k-m+1))) * u_km_*      # eq 22/23
e_km = prod_{l=m+1}^{k} 2*sqrt( (l*(2l-1)) / (2*(l+m)*(l-m)) )   # eq 21, e_mm = 1
```
`w_jkm` is the integer-overflow-sensitive one: 64-bit ints good to `k ~ 55000` (`:227`).

### 2.3 Single value `d(j,k,m,t,nB)` (`:298-371`)
`t = cos(beta)`, `nB = signbit(beta)`. Symmetry reduction to `0 <= m <= k <= j`, `beta >= 0`:
```
nB           :  d^j_{k, m}(-b) =                d^j_{m,k}( b)     (eq 5)
k<0 && m<0   :  d^j_{-k,-m}(b) = (-1)^(k-m)     d^j_{k,m}( b)     (eq 6)
m<0          :  d^j_{ k,-m}(b) = (-1)^(j+k)     d^j_{k,m}(pi-b)   (eq 7)
k<0          :  d^j_{-k, m}(b) = (-1)^(j+m)     d^j_{k,m}(pi-b)   (eq 8)
k<m          :  d^j_{ m, k}(b) = (-1)^(k-m)     d^j_{k,m}( b)     (eq 9)
j<k          :  NAN
```
Then
```
type = t>0 ? 0 : (t<0 ? 2 : 1);  tc = 1-t
c2 = sqrt((1+t)/2) ; s2 = sqrt((1-t)/2)
d^k_{k,m} = c2^(k+m) * s2^(k-m) * e_km                    (eq 18/20)
d^{k+1}_{k,m} = d^k_{k,m} * a_km_<type>                   (eq 19)
d^i_{k,m} = a_jkm_<type>(i,k,m) * d^{i-1}_{k,m} - b_jkm(i,k,m) * d^{i-2}_{k,m}   (eq 10)
```
`d(j,k,m)` (`:380-409`) is the `beta = pi/2` special case (`d^k_{k,m} = 2^{-k} e_km`).
`dSign(j,k,m)` (`:416-425`) returns the sign so that `dSign * d(j,|k|,|m|) == d(j,k,m)` at `pi/2`.
`D(j,k,m,eu)` (`:436-439`): `exp(i*(eu[0]*m + eu[2]*k)) * d(j,k,m,cos(eu[1]), signbit(eu[1]))`.

### 2.4 Tables

**(a) `dTable(jMax, t, nB, table)`** (`:452-559`) — arbitrary beta. Memory layout, `2` values per (k,m,j):
```
d^j_{k,m}(   beta)  at  table[(k*jMax*jMax + m*jMax + j)*2 + 0]
d^j_{k,m}(pi-beta)  at  table[(k*jMax*jMax + m*jMax + j)*2 + 1]
```
Size `jMax^3 * 2`. Entries with `j < max(k,m)` are uninitialized garbage. Negative `k`/`m` are recovered from
```
d^j_{-k,-m}(b) = (-1)^(k-m) d^j_{k,m}(b)
d^j_{ k,-m}(b) = (-1)^(j+k) d^j_{k,m}(pi-b)
d^j_{-k, m}(b) = (-1)^(j+m) d^j_{k,m}(pi-b)
```
Implementation loops `k in [0,jMax)`, `m in [0,k]` and fills both `(k,m)` and `(m,k)` slots (using eq-9 sign `signN = (-1)^(k-m)`; `nB` swaps `sign` and `signN`). Both `+t` and `-t` recursions run simultaneously (`a_kmFuncN`, `a_jkmFuncN` are the *other* branch). Powers of `c2` and `s2` are precomputed into a `jMax*4` work array.

**(b) `dTablePre` + `dTablePreBuild`** (`:575-691`) — same table but with precomputed `pE[k*jMax+m] = e_km`, `pW[k*jMax^2+m*jMax+i] = w_jkm(i,k,m)`, `pB[...] = b_jkm(i,k,m)` (sizes `jMax^2`, `jMax^3`, `jMax^3`). This is what the correlator uses in `derivatives()` (called once per Newton iteration).

**(c) `dTable(jMax, table, trans)`** (`:699-761`) — `d^j_{k,m}(pi/2)` only, `jMax^3` values:
```
trans == false: d^j_{k,m} at k*jMax*jMax + m*jMax + j
trans == true : d^j_{k,m} at m*jMax*jMax + k*jMax + j
```
Filled for `k >= m >= 0` and mirrored with `(-1)^(k-m)`.

### 2.5 `rotateHarmonics(bw, alm, blm, zyz)` (`:769-799`)
`b^l_m = sum_{n=-l}^{l} a^l_n D^l_{m,n}(zyz)`; `alm[n*bw + j]` layout.
```
dBeta = dTable(bw, cos(zyz[1]), signbit(zyz[1]))
for m in [0,bw):
    expAlpha = exp(i*zyz[2]*m)               # NB: uses zyz[2] (gamma) for the m phase
    for n in [0,bw):
        expGamma = exp(i*zyz[0]*n)           # NB: uses zyz[0] (alpha) for the n phase
        for j in [max(m,n), bw):
            alGamma = alm[n*bw+j]*expGamma
            vp = expAlpha*alGamma ; vc = expAlpha*conj(alGamma)
            dmn0 = dBeta[(m*bw*bw + n*bw + j)*2 + 0]
            dmn1 = dBeta[(m*bw*bw + n*bw + j)*2 + 1]
            out[m*bw+j] += vp*dmn0
            if n>0: out[m*bw+j] += vc * dmn1 * ((j+m+n)%2==0 ? 1 : -1)
```

### 2.6 Derivatives (`:814-852`)
```
csc = (nB ? -1 : 1)/sqrt(1-t^2)
d'   = d(j,k,m)*(t*k - m)*csc  -  d(j,k+1,m)*sqrt((j-k)(j+k+1))                     [0 if j==k]
d''  = d(j,k,m)*coef0 - d(j,k+1,m)*coef1 + d(j,k+2,m)*coef2
  coef0 = (t^2 k^2 + t m (1-2k) + (m^2 - k)) * csc^2
  rjk   = sqrt((j-k)(j+k+1))
  coef1 = rjk*(t*(1+2k) - 2m)*csc
  coef2 = rjk*sqrt((j-k-1)(j+k+2))
```
Symmetries listed at `:810-813` and `:833-836`.

---

## 3. Spherical cross-correlation — `include/sht/sht_xcorr.hpp` (1373 lines)

Reference: Gutman et al. (2008); normalization from Huhle et al. (2009).

### 3.1 Sizes and buffers (`:372-383`)
```
bw   = bandWidth                  (exclusive max l)
sl   = 2*bw - 1
slP  = fft::fastSize(sl)          # smallest >= sl that is a product of {2,3,5,7,11,13}
bwP  = slP/2 + 1
fm   : complex[bw*bw]             # flm * d^j_{k,m}
gn   : complex[bw]                # conj(gln) * d^j_{n,k}
fxc  : complex[slP*slP*bwP]       # half-complex 3D spectrum, layout fxc[k][n][m]
xc   : Real   [slP*slP*bwP]       # real correlation, layout xc[k][n][m], k in [0,bwP)
dBeta: Real   [bw*bw*bw*2]        # per-call wigner d(beta) table for refinement
```
`Constants` (`:361-370`): `wigD = dTable(bw, ..., trans=true)` (`bw^3`), `wigE/wigW/wigB = dTablePreBuild(bw)` (`bw^2`, `bw^3`, `bw^3`), and `plan = SepRealFFT3D(slP, Patient)`.

Strides: `fxc[k][n][m] = fxc[k*slP*bwP + n*bwP + m]`; `xc[k][n][m] = xc[k*slP*slP + n*slP + m]`.
So **the fastest axis `m` is alpha, `n` is gamma, `k` (slowest) is beta**, and the *last* axis is the half-complex one.

### 3.2 The mathematical content (naive form, `:661-687`)
```
fxc[k][n][m] = sum_{j = max(|m|,|k|,|n|)}^{bw-1}
                  fhat^j_m * conj(ghat^j_n) * d^j_{k,m}(pi/2) * d^j_{n,k}(pi/2)
```
with `k, n, m` in `[-(bw-1), bw-1]` wrapped into `[0, sl)` DFT bins, and real-signal symmetry
```
fhat^j_{-m} = conj(fhat^j_m) * (-1)^m ,   ghat^j_{-n} = conj(ghat^j_n) * (-1)^n
```
Then `xc = real 3D inverse DFT of fxc`.
This is exactly the SOFT/Gutman factorization
`D^j_{m,n}(a,b,g) = i^{n-m} sum_k d^j_{m,k}(pi/2) d^j_{k,n}(pi/2) e^{i(m a + n g - k b)}`;
the `i^{n-m}` and the sign of `b` are what produce the origin offsets in §3.4.

### 3.3 Optimised `compute()` (`:657-858`)

Flags:
```
mBw     = bw
flmFold = fNf            # z rotational order of the master; flm[m*bw+j]==0 unless m % fNf == 0
glnFold = 1              # pattern has no rotational symmetry
fMir    = fMr            # master has an equatorial mirror -> flm[m*bw+j]==0 unless (m+j)%2==0
gMir    = false
mirror  = fMir || gMir ;  bMirror = fMir && gMir   (always false here)
```
Systemic-zero masks (per column `m`, one for even `n` and one for odd `n`):
```
nonZero_p[m] = !( (m % flmFold != 0) || (bMirror && (m+p)%2 != 0) )    p in {0,1}
```
Main loop:
```
for k in [0, mBw):                                  # z slices
    # fm[m*bw + j] = flm[m*bw+j] * d^j_{k,m}(pi/2)   (wigD index m*bw*bw + k*bw + j, trans=true)
    for m in [0,mBw):  for j in [max(m,k), mBw): fm[m*bw+j] = flm[m*bw+j]*wigD[m*bw*bw+k*bw+j]

    pKpN -> fxc[ k    ][ n]     ;  nKpN -> fxc[slP-k][ n]
    pKnN -> fxc[ k    ][slP-n]  ;  nKnN -> fxc[slP-k][slP-n]

    for n in [0, bwP):                              # rows
        if (n % glnFold == 0) and (n < mBw):
            # gn[j] = conj(gln[n*bw+j]) * d^j_{n,k}(pi/2)   (wigD index k*bw*bw + n*bw + j)
            for j in [max(k,n), mBw): gn[j] = conj(gln[n*bw+j]) * wigD[k*bw*bw+n*bw+j]
            for m in [0, mBw):
                if nonZero[m]:
                    start = max(m, max(k,n))
                    if fMir and (start+m)%2 != 0: ++start
                    if gMir and (start+n)%2 != 0: ++start
                    dJ = 2 if mirror else 1
                    v = vnc = 0 ; toggle = ((start+m)%2 == 0)
                    for j in range(start, mBw, dJ):
                        vp, vc = conjMult(fm[m*bw+j], gn[j])   # (a*b, a*conj(b))
                        v   += vp
                        vnc += (vc if toggle else -vc)         # if dJ==2 sign is constant -> apply once at end
                        if dJ == 1: toggle = not toggle
                    if k % 2 != 0: vnc = -vnc
                    match = ((m+n)%2 == 0) ; s = 1 if match else -1
                    fxc[k    ][n    ][m] =  v
                    fxc[slP-k][slP-n][m] =  vnc            (only if k>0 and n>0)
                    fxc[slP-k][n    ][m] =  s*v            (only if k>0)
                    fxc[k    ][slP-n][m] =  s*vnc          (only if n>0)
                else: write zeros to the same 4 slots
        else:  zero the whole row (and its 3 mirror rows)
# zero-pad slices k in [mBw, slP-mBw]
xc = plan.inverse(fxc, dx = flmFold)
```
`conjMult` (`:1234-1243`) computes `a*b` and `a*conj(b)` sharing 4 real multiplies.

`SepRealFFT3D::inverse(spectra, signal, dx)` (`:671-678` of `util/fft.hpp`):
1. for `i = 0, dx, 2dx, ... < bwP` (i.e. skip the `m` planes that are structurally zero): 1-D backward DFTs along `k` then along `n` for that `m`-plane;
2. for all `i in [0, bwP)`: batched `c2r` along `m`, writing `signal + slP*slP*i`.
So only the first `bwP` beta-slices of the full `slP^3` cube are ever materialised. **NumPy equivalent:** build the full `(slP, slP, bwP)` half-complex array, `xc_full = np.fft.irfftn(fxc, s=(slP,slP,slP)) * slP**3`, then keep `xc_full[:bwP]`.

**Erratum (2026-08-17, `specs/2026-08-17-spherical-cross-correlation/requirements.md` D3)**: the "NumPy equivalent" above is superseded by D3's separable form -- `scipy.fft.ifft` along `k` on the `m % n_fold == 0` planes, `ifft` along `n` on the `[:bwP]` slices only, then `irfft` along `m`, all with `norm="forward"` (FFTW's unnormalised `c2r`) and `workers=1`; the full `irfftn(..., norm="forward")[:bwP]` remains a test oracle only (1.4-2.6x slower, materialises the whole `slP^3` cube).

### 3.4 Index ↔ Euler angle mapping

`indexEuler(idx, eu)` (`:580-590`), `extractInds` (`:1249-1255`) gives `knm = (k, n, m)`:
```
alpha = eu[0] = (4*m - slP) * pi / (2*slP)     =  2*pi*m/slP - pi/2
beta  = eu[1] = (2*k - slP) * pi /    slP      =  2*pi*k/slP - pi
gamma = eu[2] = (4*n - slP) * pi / (2*slP)     =  2*pi*n/slP - pi/2
```
Inverse, `eulerIndex(eu)` (`:549-575`):
```
kR = ((beta *   slP)/pi + slP)/2
nR = ((gamma*2*slP)/pi + slP)/4
mR = ((alpha*2*slP)/pi + slP)/4
if kR > slP/2:                     # glide to the stored half
    kR = slP - kR
    nR = fmod(nR + slP/2, slP)
    mR = fmod(mR + slP/2, slP)
idx = round(kR)*slP*slP + round(nR)*slP + round(mR)
```
The glide encodes the ZYZ identity `R(a, b, g) == R(a+pi, -b, g+pi)`.

**Erratum (2026-08-17, `specs/2026-08-17-spherical-cross-correlation/requirements.md` D6)**: `eulerIndex` has no caller in EMSphInx, no `beta` wrap (an unwrapped `beta`, e.g. `pi + 0.1` at `slP` 135, gives `kR = 137.15 -> slP - kR = -2.15`, a `size_t` wrap), and a corner for odd `slP` at `beta = 0 (mod 2 pi)`: `kR = slP/2` is not glided (`>` is false) and rounds to `bwP`, outside the stored half. The port `euler_to_index` wraps `beta` with `_euler.wrap_beta`, reduces `alpha`/`gamma` into `[-pi/2, 3 pi/2)` before the formulas, rounds with `floor(x + 0.5)` (`std::round`, not banker's), reduces `n`, `m` modulo `slP` and clamps `k` to `bwP - 1`.

`extractNeighborhood<N>(idx, nh)` (`:505-544`) — `(2N+1)^3` window with **periodic** wrap in all three axes, then the glide is applied to any sampled `k >= bwP`:
```
if inds_k[i] >= bwP:
    inds_m[i] = inds_m[i] + bwP - 1  if inds_m[i] < bwP  else  inds_m[i] - bwP
    inds_n[i] = same
    inds_k[i] = slP - inds_k[i]
nh[k][n][m] = xc[inds_k[k]*slP*slP + inds_n[n]*slP + inds_m[m]]
```

**Erratum (2026-08-17, `specs/2026-08-17-spherical-cross-correlation/requirements.md` D5)**: the C++ `extractNeighborhood` (`:527-533`) has *two* defects, both reproduced by the port under `emsphinx_compatible=True` and pinned by tests: (i) the alpha/gamma shift is applied **per slot index `i`** (`inds[2][i]`, `inds[1][i]`), not per glided *plane*, so when the `k + 1` (or `k - 1`) slot is glided the `n = i`/`m = i` slots of all three planes are shifted while the other slots of the glided plane are not; (ii) for even `slP` the shift `x < bwP ? x + bwP - 1 : x - bwP` (section 8 item 12) is applied identically to the alpha slot `:530` **and the gamma slot `:531`**, and at `x = bwP - 1 = slP/2` yields `slP`, one past the axis -- an `m` slot reads the next row's first element, an `n` slot the next beta slice's row 0 (silently in-buffer), and at `k0 = bwP - 1` the shifted `n` slot reaches **past the end of `xc`** (`(bwP - 1, bwP - 2, any m0)` and three more triples; undefined behaviour, clamped to `xc.flat[-1]` in the port and never asserted against the C++). `emsphinx_compatible=False` uses the per-plane glide (exact for even `slP`, the half-cell approximation `s = (slP - 1)/2` for odd).

`extractBunge(zxz)` (`:595-649`) — repacks `xc` into a ZXZ (Bunge) cube with origin at 0, `phi1` fastest, `phi2` middle, `Phi` slowest (`ZYZ -> ZXZ` is `phi1 = alpha - pi/2`, `Phi = beta`, `phi2 = gamma + pi/2`). Contains a documented half-pixel shift.

**Erratum (2026-08-16, `specs/2026-08-16-sht-wigner-d/requirements.md`)**: the `ZYZ -> ZXZ` offsets quoted above are EMSphInx' reversed `zyz2eu` relation (`rotations.hpp:1025-1039`) as `extractBunge` uses it; the relation consistent with `zyz2qu` and with `test/xtal/rotations.cpp:296-310` is `phi1 = alpha + pi/2`, `Phi = beta`, `phi2 = gamma - pi/2` (`_euler.zyz_to_bunge`, Phase 3). A port of `extractBunge` (Phase 4/9) must use `_euler.bunge_to_zyz` and record the deviation.

### 3.5 Peak finding

`findPeak()` (`:862-876`): plain argmax over the whole `xc` array (`slP*slP*bwP` entries).

`interpPeak(ind0, eu)` (`:406-432`):
```
knm  = extractInds(ind0, slP)
xCorr = extractNeighborhood<1>(ind0)          # 3x3x3
x = [0,0,0]
peak0 = interpolateMaxima(xCorr, x)           # x = (dk, dn, dm) subpixel offset in [-1,1]
xMax = max(|x[0]|, max(|x[1]|, |x[0]|))       # *** BUG: x[2] never checked, x[0] twice ***
if xMax > 1: x = 0 ; peak0 = xCorr[1][1][1]
alpha = ((m + x[2])*4 - slP)*pi/(2*slP)
beta  = ((k + x[0])*2 - slP)*pi/(   slP)
gamma = ((n + x[1])*4 - slP)*pi/(2*slP)
```

`detail::interpolateMaxima(p[3][3][3], x)` (`:1261-1366`): fits the **full tri-quadratic tensor product**
`f(x,y,z) = sum_{i,j,k in {0,1,2}} a_{kji} x^i y^j z^k` (27 coefficients, exact through the 27 samples;
closed-form coefficient expressions at `:1265-1308`), then runs Newton's method (max 25 iterations,
`eps = sqrt(machine eps)`) on the 3x3 analytic gradient/Hessian; if not converged, `x` is reset to `0`.
Returns the fitted value at the located maximum. Array ordering is `p[z][y][x] = p[k][n][m]`.

### 3.6 `refinePeak` — Newton refinement on the continuous correlation (`:442-499`)
```
eu0     = copy of eu
absEps  = eps * 2*pi / slP                 # eps default 0.01 -> ~1% of a grid cell
euEps   = sqrt(machine eps)
maxIter = 15
prevMag2 = 2*pi*3/slP                      # first step must be < ~1 pixel per axis
for iter in 1..maxIter:
    peak = derivatives(flm, gln, eu, jac, hes, bw, fMr, fNf)
    try:
        if isnan(hes[4]): raise                       # beta degeneracy
        solve::cholesky(hes, step, jac, 3)            # fails (throws) on non-PD -> avoids saddles
        mag2 = |step|^2 ; if mag2 > prevMag2: raise   # steps must shrink
        prevMag2 = mag2
    except:
        if isnan(hes[4]):                             # exactly on beta = 0 or +/- pi
            step = [jac[0]/hes[0], 0, 0]              # 1x1 sub-problem
        else:                                         # near degeneracy: 2x2 sub-problem
            det = hes[0]*hes[4] - hes[1]*hes[1]
            if |det| < euEps: raise "singular matrix"
            if det  < euEps: raise "converging to saddle"
            step = [(jac[0]*hes[4] - jac[1]*hes[1])/det,
                    (jac[1]*hes[0] - jac[0]*hes[1])/det, 0]
    eu -= step
    if max(|step|) < absEps: break
    if iter == maxIter: raise "failed to converge"
on any exception: eu = eu0 ; return derivatives(..., der=false)
```

### 3.7 `derivatives()` — direct XC + Jacobian + Hessian at one rotation (`:889-1119`)

Wraps `beta` into `[-pi, pi]`, sets `t = cos(beta)`, `nB = signbit(beta)`,
`csc = (nB?-1:1)/sqrt(1-t^2)`, then `dTablePre(mBW, t, nB, dBeta, wigE, wigW, wigB)`.
`exp(i*m*alpha)` and `exp(i*n*gamma)` are generated by Chebyshev `T_n`/`U_n` recursions.
`wrk[10] = {xc, d/da, d/db, d/dg, d2/da2, d2/db2, d2/dg2, d2/dadb, d2/dbdg, d2/dgda}`.
For each `(m, n)` with `m,n >= 0`, `dJ = 2 if fMir else 1`, `start = max(m,n)` (+1 if `fMir && (start+m)%2`):
```
agP, agN = conjMult(expAlpha, expGamma)   # exp(i(m a + n g)), exp(i(m a - n g))
sign = (-1)^(n+m) ; sn = (-1)^n
agP *= sign ; agN *= sign*sn
for j in range(start, mBw, dJ):
    d0P = dBeta[(m*mBW^2 + n*mBW + j)*2 + 0]   # d^j_{m, n}(beta)
    d0N = dBeta[(m*mBW^2 + n*mBW + j)*2 + 1]   # d^j_{m, n}(pi-beta)
    (+ d^j_{m+1,n}, d^j_{m+2,n} for the derivative terms)
    d1P = d0P*coef1_0PP - d0P_1*rjm                          # dd/dbeta for +n
    d1N = d0N*coef1_0PN + d0N_1*rjm                          # dd/dbeta for -n
    d2P = d0P*coef2_0PP - d0P_1*rjm*coef2_1PP + d0P_2*coef2_2
    d2N = d0N*coef2_0PN + d0N_1*rjm*coef2_1PN + d0N_2*coef2_2
    vp, vc = conjMult(flm[m*bw+j], gln[n*bw+j]) ; if (j+m)%2: vp = -vp
    vcPP = vc*agP ; vpPN = vp*agN
    contributions:  Re(vcPP*d0P) for (+m,+n) and (-m,-n)
                    Re(vpPN*d0N) for (+m,-n) and (-m,+n)
    d/da = Im(...)*(-m), d/dg = Im(...)*(-n or +n), d/db = Re(v*d1)
    d2/da2 = Re*(-m^2), d2/dg2 = Re*(-n^2), d2/db2 = Re(v*d2)
    d2/dadb = Im(v*d1)*(-m), d2/dbdg = Im(v*d1)*(-/+n), d2/dgda = Re*( -/+ m n )
accumulate ×1 for (+m,+n); ×1 for (+m,-n) if n>0; ×1 for (-m,+n) and (-m,-n) if m>0
```
Hessian is symmetrised: `hes = [[w4, w7, w9],[w7, w5, w8],[w9, w8, w6]]`.

Coefficient definitions (`:1009-1020`):
```
coef2_0a  = t^2*m^2 + (n^2 - m)
coef2_0b  = t*n*(1 - 2m)
coef2_1a  = t*(1 + 2m)
coef1_0PP = ( t*m - n)*csc          coef1_0PN = ( t*m + n)*csc
coef2_0PP = (coef2_0a + coef2_0b)*csc^2   coef2_0PN = (coef2_0a - coef2_0b)*csc^2
coef2_1PP = (coef2_1a - 2n)*csc     coef2_1PN = (coef2_1a + 2n)*csc
rjm       = sqrt((j-m)(j+m+1)) ;   coef2_2 = sqrt((j-m-1)(j+m+2))*rjm  (0 if j==m)
```

### 3.8 `correlate()` (`:394-400`)
```
compute(flm, gln, fMr, fNf, xc)
ind0 = findPeak()
peak = interpPeak(ind0, eu)
return refinePeak(flm, gln, fMr, fNf, eu, eps) if ref else peak
```

### 3.9 Normalized correlator (`:236-270`, `:1128-1225`)

`UnNormalizedCorrelator` just stores `flm`, `fMr`, `fNf`.

`NormalizedCorrelator(bw, flm, flm2, fMr, fNf, mlm)` where
`flm2 = SHT(f^2)` and `mlm = SHT(binary mask covering the detector footprint)`.
Constants (`:1182-1204`):
```
rDen  = compute(flm , mlm)                  # mrf  : mask (x) reference,     grid of slP*slP*bwP
tmp   = compute(flm2, mlm)                  # mrf2 : mask (x) reference^2
s2m   = mlm[0].real() * sqrt(4*pi)          # integral of the window (binary mask, [0,4pi])
fWbar = mrf / s2m                                                     # Huhle eq 9
rDen  = 1 / sqrt( mrf2 - 2*fWbar*mrf + fWbar^2*s2m )                  # Huhle eq 8
```
`correlate(gln, eu, ref, eps)` (`:1140-1159`):
```
compute(flm, gln, mr, nf, xc)
single pass: xc[i] *= rDen[i]; track argmax
peak = interpPeak(iMax, eu)
return refinePeak(gln, eu, eps) if ref else peak
```
`refinePeak` (`:1169-1172`): unnormalized Newton refine, then divide by `denominator(eu)` evaluated on the fly via two `derivatives(..., der=false)` calls. Note the documented caveat (`:263-264`): the chain rule through the moving window is *not* applied.

### 3.10 The confidence value ('xc'/'ci')
`Result::corr` is the value returned by `PhaseCorrelator::correlate` — i.e. the raw (FFT-unnormalized) correlation, optionally divided by the Huhle denominator. It is **not** further divided by the pattern's standard deviation: `Indexer::computeHarmonics` stores `sum2 = prj->unproject(...)` (which returns `sqrt(omgW/omgS*4*pi)`) but `Indexer::correlate` never uses it (`idx/indexer.hpp:315, 326-331`). Since the back-projected pattern is already forced to zero-mean/unit-stdev inside the window, `sum2` is a per-geometry constant. Consequence: the metric is comparable across patterns and phases for a fixed setup, but is not an absolute `[0,1]` NCC.
`Result::iq` is a separate DCT-based image-quality metric (§5.4).

---

## 4. Detector geometry and back-projection — `include/modality/ebsd/detector.hpp`

### 4.1 `Geometry<Real>` fields (`:53-65`)
```
dTlt, dOmg   detector tilt / rotation (degrees)      [dOmg must be 0]
sTlt, sOmg   sample   tilt / rotation (degrees)      [sOmg must be 0]
w, h         camera width/height in pixels
pX, pY       pixel width/height in microns
cX, cY       pattern center relative to the IMAGE CENTER, in (fractional) pixels
sDst         scintillator distance in microns
circ         apply circular mask
flip         images are vertically flipped (image origin top-left vs cartesian bottom-left)
```
All zero-initialised by the default ctor.

### 4.2 Pattern-center conventions (`:249-279`)
The internal representation is **EMsoft**: `cX = xpc`, `cY = ypc` (pixels from image centre), `sDst = L` (microns).
```
patternCenter      (xpc,ypc,L)     : cX=xpc ; cY=ypc ; sDst=L
patternCenterTSL   (x*,y*,z*)      : cX = x*·w - w/2 ; cY = y*·w - h/2 ; sDst = z*·w·pX
patternCenterOxford(x*,y*,z*)      : cX = (x*-0.5)·w ; cY = (y*-0.5)·h ; sDst = z*·w·pX
patternCenterBruker(x*,y*,z*)      : cX = (x*-0.5)·w ; cY = (0.5-y*)·h ; sDst = z*·h·pX
```
(`camera size and pixel size must be set first`; namelist `vendor` selects which, `modality/ebsd/idx.hpp:221-229`; accepted strings `EMsoft`, `EDAX`, `tsl`, `Oxford`, `Bruker`).

`ecp(dim, theta)` (`:285-298`) for electron-channeling: `sTlt=sOmg=0`, `dTlt=-90`, `cX=cY=0`, `circ=true`, `flip=false`, `w=h=dim`, `sDst=10000`, `pX=pY=tan(theta*pi/(dim*90))*sDst`.

`bin(n)` (`:304-326`): `w,h /= n` (must divide exactly), `pX,pY *= n`, `cX,cY /= n`.
`rescale(wNew,hNew)` (`:432-450`): `sx = w/wNew`, `sy = h/hNew`, `pX *= sx`, `pY *= sy`, `cX /= sx`, `cY /= sy` (same solid angle). *Note* `rescale(scale)` (`:421-426`) calls `rescale(wNew, wNew)` — passes width twice; a latent bug (unused by the EBSD path, which calls the two-argument form).
`readEMsoft(grp)` (`:475-495`) reads `thetac, delta, xpc, ypc, L, numsx, numsy, binning, maskpattern`.

### 4.3 Direction → detector pixel: `interpolatePixel(n, pix, flp)` (`:334-373`)
```
if n[2] < 0 : return false                                   # can't project through the sample
if dOmg != 0 or sOmg != 0: throw "omega tilt not yet supported"
alpha = (90 - sTlt + dTlt) * pi/180                          # detector-vs-sample angle
if |alpha| > pi/2: throw "pattern center not on detector"
sA = sin(alpha) ; cA = sqrt(1 - sA^2)
d  = sDst / (n[0]*sA + n[2]*cA)                              # distance along n to detector plane
if d < 0: return false
x = n[1] * d                                                 # microns, detector-x
y = (sA*n[2] - cA*n[0]) * d                                  # microns, detector-y
X = (cX + x/pX)/w + 0.5                                      # fractional [0,1]
Y = (cY + y/pY)/h + 0.5
if X<0 or Y<0 or X>1 or Y>1: return false                    # off frame
if flp: Y = 1 - Y                                            # vertical flip AFTER bounds test
if circ:                                                     # circular mask
    dX = (X-0.5)*w ; dY = (Y-0.5)*h ; r = min(w,h)/2
    if r*r < dX*dX + dY*dY: return false
pix->bilinearCoeff(X, Y, w, h)
return true
```
Inverse `sampleDir(X, Y, n)` (`:380-395`), `X,Y` fractional relative to the detector:
```
alpha = (90 - sTlt + dTlt)*deg2rad ; sA=sin ; cA=sqrt(1-sA^2)
fx = (cX - X*w)*pX ;  fy = (cY - Y*h)*pY ;  den = sqrt(sDst^2 + fx^2 + fy^2)
n = [ (sDst*sA + fy*cA)/den , -fx/den , (sDst*cA - fy*sA)/den ]
```
Consequence for a standard EBSD setup (`sTlt=70`, `dTlt=10` → `alpha = 30 deg`): the pattern centre maps to
`n = (sin alpha, 0, cos alpha) = (0.5, 0, 0.866)`, and the whole detector footprint lies in the **northern** hemisphere.

`solidAngle(gridRes)` (`:401-415`): fraction of a `(gridRes+1)^2` square-Lambert north-hemisphere sampling that lands on the detector, divided by `gridRes^2 + (gridRes-2)^2`.
`scaleFactor(dim)` (`:465-469`): `sqrt( solidAngle(501) * (2*dim^2 - 4*(dim-1)) / (w*h) )` — makes the average detector pixel the same angular size as an average spherical-grid pixel.
`northPoleQuat()` (`:455-459`): **currently returns identity `{1,0,0,0}`**; the intended `{cos(theta/2), 0, sin(theta/2)*pijk, 0}` with `theta = (90 - sTlt + dTlt)*pi/180` is commented out.

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D2)**: (i) `bilinearCoeff` maps the fractional position `X in [0, 1]` to `x = X (w - 1)`, i.e. it treats the pixel centres `0 .. w-1` as spanning the *whole* physical width -- pixel `c` (physical centre `(c + 0.5)/w`) is sampled at `c + 0.5 - (c + 0.5)/w`, a stretch of up to `+-0.49` px at the edges (measured `0.497 / 0.489` px in the 49-px resampled image at `bw` 68). Phase 5 replaces `interpolatePixel` by the exact inverse of kikuchipy's `_get_direction_cosines_for_fixed_pc` (pixel-centre convention, `1.6e-14` px) and maps into the resampled image with the DCT sampling convention `(col + 0.5) w_out/ncols - 0.5`; the stretch is **not reproduced** (measured effect on the forward-projection lock at `bw` 68: normalised median 0.424 -> 0.488 deg, inside the coarse tolerance either way, so an `IndexEBSD` parity run cannot tell them apart; pinned structurally in the LUT test with the stretch as a negative control). (ii) `patternCenterBruker` sets `cX = (x* - 0.5) w` while the EMsoft branch `patternCenter` sets `cX = xpc`, so EMSphInx equates `xpc = (x*_B - 0.5) w` -- kikuchipy's *pre-v5* EMsoft relation (`_pc_emsoft2bruker`, `version < 5`; v5 has `xpc = (0.5 - x*_B) N b`). Namelists for `IndexEBSD` (Phases 9/10) must use `vendor = 'Bruker'` with kikuchipy's `pc`, never `vendor = 'EMsoft'` with `pc_emsoft()`; the pattern file must be one EMSphInx reads with `flip = true` or be written vertically flipped.

### 4.4 `BackProjector::Constants` — build the interpolation LUT (`:502-569`)
```
sclr = image::Rescaler(geo.w, geo.h, geo.scaleFactor(dim) * fct, Patient)   # fct = sqrt(2) from the indexer
xyz       = square::legendre::normals(dim)                # north hemisphere directions
omegaRing = square::solidAngles(dim, Layout::Legendre)    # per-ring relative pixel size
g         = geo.rescale(sclr.wOut, sclr.hOut)             # geometry of the resampled detector

# north hemisphere
for i in [0, dim*dim):
    n = qu ? rotateVector(qu, xyz[i]) : xyz[i]
    if g.interpolatePixel(n, &p, flip):
        p.idx = i ; iPts.push_back(p) ; omeg.push_back(omegaRing[ringNum(dim, i)])

# south hemisphere (skip the equator ring = the 4 border rows/columns)
for i in [0, dim*dim):
    y = i/dim ; if y==0 or y==dim-1: continue
    x = i-dim*y ; if x==0 or x==dim-1: continue
    xyz[3i+2] = -xyz[3i+2]
    n = qu ? rotateVector(qu, xyz[i]) : xyz[i]
    if g.interpolatePixel(n, &p, flip):
        p.idx = i                      # <<< NOT dim*dim + i  -- see gotchas
        iPts.push_back(p) ; omeg.push_back(omegaRing[ringNum(dim, i)])

omgW = sum(omeg)                       # solid angle covered by the window
omgS = sum over the whole sphere of omegaRing[ringNum(...)], counting equator pixels once
```
`image::BiPix::bilinearCoeff(X, Y, w, h)` (`util/image.hpp:526-551`):
```
x = X*(w-1) ; y = Y*(h-1)
i0 = min(int(x), w-1) ; j0 = min(int(y), h-1) ; i1 = min(i0+1, w-1) ; j1 = min(j0+1, h-1)
inds = [j0*w+i0, j0*w+i1, j1*w+i0, j1*w+i1]
wx1 = x-i0 ; wy1 = y-j0 ; wx0 = 1-wx1 ; wy0 = 1-wy1
wgts = [wy0*wx0, wy0*wx1, wy1*wx0, wy1*wx1]
```

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D1, D3, D4)**: (i) the south loop is **unreachable**: `interpolatePixel` rejects every `n[2] < 0` at `:336` before anything else, so section 8 item 9 stays latent -- the port gathers only the north grid (physical guard `z_s >= 0` on the sample-frame normal, which is the sphere frame) and returns an all-zero south hemisphere; a footprint below the sample plane is clipped (measured: `sample_tilt` 0 keeps 40 % of the `sample_tilt` 70 window). (ii) `omgS == 2 dim^2 - 4 (dim - 1)` exactly (measured `5834 / 9802 / 16202` at `dim` 55 / 71 / 91) because `omegaRing` is relative to the average pixel. (iii) `solidAngle(501)` (`:401-415`) evaluates `(gridRes + 1)^2 = 252004` northern points but divides by `gridRes^2 + (gridRes - 2)^2 = 500002`; the count consistent with the loop is `502^2 + 500^2 = 502004`, so the fraction is biased 0.4 % high (Ni detector: `0.124350` vs `0.123854` with the circle). Ported literally and labelled a quirk, because the `Rescaler` sizes must match `IndexEBSD` for the DCT resample to be the same operation. (iv) The C++ applies **two circles**: `solidAngle` runs `interpolatePixel` on the *unrescaled* geometry (`r = min(w, h)/2` about the physical centre) while the LUT loop runs it on `g = geo.rescale(wOut, hOut)` (`r = min(wOut, hOut)/2` in rescaled pixels), which is an ellipse in detector pixels for rectangular detectors because `wOut`/`hOut` round independently (`(48, 60)` at `bw` 68: 953 physical vs 958 C++ points, 5 differ). The port uses the physical circle once. (v) `Constants` calls the two-argument `rescale` (`:511`) and inherits no empty-geometry guard: a footprint that covers no part of the northern hemisphere gives `wOut = hOut = 0` and fails inside FFTW; the port raises `ValueError` for `rescaled_shape < 1 px` and again for an empty LUT (`n_points == 0`).

### 4.5 `BackProjector::unproject(pat, sph, iq)` (`:589-623`)
```
vIq = sclr.scale(pat, rPat, sWrk, zer=true, flt=0, iq=(iq != NULL))   # DCT-based resample + zero mean
if iq: *iq = vIq
for i: iVal[i] = iPts[i].interpolate(rPat)                       # bilinear from resampled detector
mean  = sum(iVal[i]*omeg[i]) / omgW                              # solid-angle weighted mean
iVal -= mean
stdev = sqrt( sum(iVal[i]^2 * omeg[i]) / omgW )
if stdev == 0:
    sph[iPts[i].idx] = 1 for all i ;  return 0                   # used to build the binary window mask
else:
    iVal /= stdev
    sph[iPts[i].idx] = iVal[i]
    return sqrt( omgW/omgS * 4*pi )                              # 'var', a per-geometry constant
```
`mask(sph)` (`:628-630`): sets `sph[p.idx] = 1` for every LUT point.

**`sph` is never cleared between patterns** — it is allocated zeroed once in the `Indexer` (`idx/indexer.hpp:169`) and only the covered points are overwritten, so uncovered spherical pixels stay at exactly `0` (the correct "no data" value given the zero-mean normalisation).

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D4, D5)**, a porting note (ours, not EMSphInx'): pocketfft's `scipy.fft.dctn(type=2)` of a constant image is not an exact delta -- AC coefficients up to `1.1e-11` for `np.full((60, 60), 37.0)` -- so a constant pattern pushed through the literal chain gives a resampled image of amplitude `~4e-11`, a weighted `stdev` of `5.4e-12 != 0`, and the `stdev == 0` branch above is **not** taken; the sphere would receive rounding noise normalised to unit variance. `IndexEBSD` builds its window mask through exactly this path (`unproject(win = ones)`, `idx.hpp:266-268`), presumably relying on FFTW's `REDFT10` being exact for a constant. The port builds the mask directly (`BackProjector::mask` semantics), tests `np.ptp(pattern) == 0` before the DCT and takes the mask branch when true (`iq = 1.0` for a non-zero constant, `0.0` for all-zero -- the exact `imageQuality` values), and keeps the literal `stdev == 0.0` branch in the kernel (reachable from an all-zero resampled image; a `stdev <= tiny` relaxation is guarded against by a test with an image of amplitude `1e-14`, which must be normalised).

### 4.6 `image::Rescaler` (`util/image.hpp:143-186, 564-619`)
Resampling is done by **2-D DCT-II → truncate/zero-pad in the frequency domain → DCT-III**:
```
fwd->execute(in, wrk)                       # FFTW_REDFT10 in both axes
vIq = imageQuality(wrk, wIn, hIn)           # optional, on the DCT of the ORIGINAL size
truncate/pad rows to wOut, rows to hOut with zeros
if zer: wrk[0] = 0                          # DC -> 0 (zero mean)
optional high-pass: for j,i < flt with r = sqrt(i^2+j^2) <= flt and (i>0 or j>0):
                     wrk *= cos((r/(2*flt) + 0.5)*pi)^2
inv->execute(wrk, out)                      # FFTW_REDFT01
```
FFTW's `REDFT10`/`REDFT01` pair is unnormalised by `2*N` per axis; that scale factor is **not** removed here (it cancels because the pattern is subsequently normalised to unit stdev). `MasterPattern::resize` does apply an explicit correction `0.5/nhScaled.size()` (`idx/master.hpp:365-367`).

---

## 5. Pattern preprocessing

### 5.1 `PatternProcessor` — `include/modality/ebsd/imprc.hpp`
`setSize(w, h, r = circRad, b = gausBckg, n = nRegions)` (`:107-146`):
```
nPix = w*h
doBkg = b
r == -1  -> msk=false, bkg = BckgSub2D(w,h)                  # all-ones mask
r ==  0  -> msk=true , bkg = BckgSub2D::CircMask(w,h)        # radius = min(w,h)/2
r >  0   -> msk=true , bkg = BckgSub2D::CircMask(w,h,r)
doAhe = (n > 0) -> ahe.setSize(w, h, n, n)
```
`process(TPix im, Real* buf)` (`:166-191`) — **order of operations**:
```
if doBkg:
    bkg.fit(im) ; bkg.subtract(im, buf)               # out of place, float
    if doAhe:
        rescale buf linearly to uint8 [0,255] using its own min/max -> work
        ahe.equalize(work, buf, msk ? bkg mask : NULL)
elif doAhe:
    if TPix is uint8: ahe.equalize(im, buf, mask)
    else: rescale im to uint8 with its own min/max -> work ; ahe.equalize(work, buf, mask)
else:
    buf = (Real)im
```
Note the background subtractor is **always** constructed (even for `r == -1`) because it owns the mask used by AHE.

### 5.2 2-D Gaussian background — `include/util/gaussian.hpp`
Model: `f(x) = c*exp(-(x-a)^2/b)` (so `b = 2*sigma^2`), `Model::evaluate` (`:128-133`).
`estimate` (`:140-163`): `c = max(y)`, `a = argmax`, then `b = xy/y2` from a log-linear regression of `ln(y/c) = -(x-a)^2/b`.
`fit` (`:172-231`): Gauss–Newton on `(a,b,c)`, `maxIter = 50`, convergence when `|(ssPrev-ss)/ss| < 1e-4` and non-decreasing; solved with `solve::cholesky(jTj, step, jTr, 3)`; returns `R^2`; throws on non-convergence.
`BckgSub2D::CircMask(w,h,r)` (`:238-251`): `mask[j*w+i] = 1` iff `(i-w/2)^2 + (j-h/2)^2 <= r^2`.
`BckgSub2D::fit(im)` (`:257-316`):
```
rWrk[j] = max over masked pixels of row j        (init to im[0])
cWrk[i] = max over masked pixels of col i
gx.fit(NULL, &cWrk[1], w-2)      # on failure: a = w/2, b = inf, c = mean(cWrk)
gy.fit(NULL, &rWrk[1], h-2)      # on failure: a = h/2, b = inf, c = mean(rWrk)
c = max(gx.c, gy.c)
cWrk[i] = exp(-(gx.a - i)^2 / gx.b) * c
rWrk[j] = exp(-(gy.a - j)^2 / gy.b)
```
`subtract(TPix im, Real* bf)` (`:379-386`): `bf[j*w+i] = mask ? im - rWrk[j]*cWrk[i] : 0`.
The integer in-place variants (`:335-372`) add `offset = c/2` back and clamp to the type range; masked-out pixels are set to `round(offset)`.

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D9)**: (i) the stopping rule `|(ssPrev - ss)/ss| < 1e-4` and non-decreasing has a hole: an exact Gaussian with an *integer* mean (e.g. `(a, b, c) = (30, 128, 200)`) is estimated exactly, `ss` is `0` at every iteration, the metric is `0/0 = NaN`, no comparison is ever true and `fit` throws non-convergence -> `BckgSub2D::fit` falls back to the flat background (`a = w/2`, `b = inf`, `c = mean`); reproduced by the port (`error_model="numpy"` on `_fit_gaussian_1d_kernel`, the second sanctioned kernel after Phase 4's `_interpolate_maxima`). Irrelevant for real patterns, which always have residuals. (ii) `rWrk`/`cWrk` are initialised to `im[0]` (`:259-260`) **even when pixel `[0, 0]` is masked out**; faithful. (iii) NaN semantics of `solve::cholesky` (`linalg.hpp:411-431`): both tests (`signbit(a[i, i]) != neg`, `real(sum) < eps`) are *false* for NaN, so a flat input (`b = 0/0 = NaN` in `estimate`, an all-NaN `jTj`) never fails the decomposition -- all 50 iterations run with NaN steps and the fit reports non-convergence, exactly the C++ path. The port writes the comparisons in the C++ direction (`if pivot < eps`, never `if not (pivot >= eps)`); the status on the flat input is platform-dependent and recorded, not asserted.

### 5.3 AHE — `include/util/ahe.hpp` (`AdaptiveHistogramEqualizer<Real, uint8_t>`)
```
HistBins = 256 for uint8
setSize(w, h, nx, ny):
    tx = w/nx ; ty = h/ny ; hWdth = 0.5   # 0.5 = mosaic (non-overlapping tiles); 1.0 = 50% overlap
    tile (i,j) centre  midX = tx*i + tx/2 , midY = ty*j + ty/2   (rounded)
    tile bounds        [round(mid - t*hWdth), round(mid + t*hWdth)] clamped to [0,w] / [0,h]
    per-row/col interpolation pairs (l, u, c, f):
       u = upper_bound(mids, index)
       if u == n:  l = u = n-1 ; c = f = 0.5      # past last centre
       elif u == 0: l = u = 0  ; c = f = 0.5      # before first centre
       else: l=u-1 ; f = (idx - mid[u-1])/(mid[u]-mid[u-1]) ; c = 1-f
computeHist(im, msk):
    per tile: histogram over its bounds (skipping msk==0 pixels)
              if all pixels masked out -> fill histogram with 1 (identity ramp)
              partial_sum -> CDF ; scale by (HistBins-1)/CDF[last]
equalize(im, buf, msk):
    buf[p] = cdf[l_j + l_i][v]*c_j*c_i + cdf[l_j + u_i][v]*c_j*f_i
           + cdf[u_j + l_i][v]*f_j*c_i + cdf[u_j + u_i][v]*f_j*f_i
    (the in-place uint8 variant adds 0.5 before truncating)
```
(There is a second, independent implementation `image::adHistEq` in `util/image.hpp:285-417` used elsewhere; it works on floats, has NaN handling, and uses a *streaming* row-of-tiles scheme. The pattern pipeline uses the `ahe.hpp` class.)

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D9)**: (i) **masked-out pixels are equalised too** -- the mask only affects the histograms (measured: the circle mask changes a Ni pattern by up to 164.6 gray levels and the masked-out corner pixels come out in `[0, 137]`); (ii) a **uniform image maps to 255 everywhere** (the CDF is a step at the value; asserted in the port so nobody "fixes" it to identity); (iii) for `n_regions` that divide the shape the mosaic AHE equals `skimage.exposure.equalize_adapthist(p, kernel_size=shape // n_regions, clip_limit=0, nbins=256) * 255` to within 6 gray levels (max 5.90, mean 2.99, correlation 1.000000 on the Ni patterns; kikuchipy's `adaptive_histogram_equalization(kernel_size=(6, 6), clip_limit=0, nbins=256)` to 1.02 after its min-max rescale); for non-dividing tiles they diverge (`n_regions` 7 on 60 px: max 70, correlation 0.924 -- `skimage` pads to a multiple of the kernel).

### 5.4 Image quality (`util/image.hpp:489-507`)
```
imageQuality(dct, w, h):
   vIq  = sum_{i,j} |dct[j][i]| * (i^2 + j^2)
   sumP = sum |dct|
   sumW = sum (i^2+j^2)
   return sumP == 0 ? 0 : 1 - vIq / (sumW*sumP/(w*h))
```
Computed on the DCT-II of the *raw* pattern inside `Rescaler::scale` (so before resampling but after preprocessing).

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D5)**: the statistic is **offset-dependent** because the DC term enters `sumP` (measured on Ni patterns: `+100` on the pattern moves the IQ by `+0.056`; `0.7663-0.7788` over the nine `nickel_ebsd_small` patterns) and scale-invariant; a pure-DC spectrum gives exactly `1.0`, the all-zero pattern `0.0` (the `sumP == 0` branch). It correlates 0.62 with kikuchipy's `get_image_quality` (Krieger Lassen, FFT power spectrum of the normalised pattern) -- a different statistic, documented, not reconciled.

### 5.5 Pattern files — `include/modality/ebsd/pattern.hpp`
`PatternFile : ImageSource` with `numPat()`, `flipY()`, `extract(out, cnt)` (thread-safe under a mutex, returns the vector of pattern indices actually pulled). Vertical flip flags:
```
up1/up2 (EDAX)              -> flp = true    (:450)
HDF5, vendor "EDAX"         -> true
HDF5, vendor "EMsoft"       -> true
HDF5, "Oxford"/"Bruker"/"Bruker Nano"/"DREAM.3D" -> false
raw *.data float            -> false  (:533)
Oxford *.ebsp               -> false  (:857)
```
Pixel types: `U8`, `U16`, `F32` (`idx/base.hpp:50-55`).

---

## 6. The indexer

### 6.1 Abstract interfaces — `include/idx/base.hpp`
- `ImageSource` (`:46-92`): `pixelType()`, `width/height/numPix/imBytes`, `virtual extract(out, cnt)`.
- `ImageProcessor<Real>` (`:96-118`): `process(uint8_t/uint16_t/float const*, Real*)`, `numPix()`, `clone()`.
- `BackProjector<Real>` (`:122-141`): `unproject(im, sph, iq)` returning `sqrt(int im^2 dOmega)`, `northPoleQuat()` (default identity), `clone()`.

### 6.2 `Indexer<Real>` — `include/idx/indexer.hpp`
Members (`:68-85`):
```
mBw  = bw
dim  = mBw + (mBw%2 == 0 ? 3 : 2)     # smallest odd side length whose Legendre bw >= mBw
quNp = backPrj->northPoleQuat()
sum2                                   # set by unproject, never consumed
pSym[phase]                            # pseudo-symmetric quaternions per phase (default empty)
wrk[imPrc->numPix()] , sph[dim*dim*2] , gln[mBw*mBw]
prc, prj, sht = DiscreteSHT(dim, dim-2, Legendre), xc[phase]
```
`Result<Real>` (`:54-64`): `{corr, iq, phase, qu[4]}`; `operator<` sorts **descending** by `corr`.

`computeHarmonics(im)` (`:312-318`):
```
prc->process(im, wrk)
sum2 = prj->unproject(wrk, sph, &iq)
sht.analyze(sph, gln, mBw, mBw)        # north = sph, south = sph + dim*dim
return iq
```

`indexImage(pat, res, n, ref)` (`:216-270`):
```
for i in [0,n): res[i] = {corr:0, phase:-1, qu:0}
iq = computeHarmonics(pat)
for p in phases:
    r = {corr: xc[p]->correlate(gln, r.qu, ref), phase: p, iq: iq}   # r.qu[0..2] holds ZYZ
    insert r into the sorted top-n list (upper_bound + shift down)
    q0 = zyz2qu(r.qu)
    for q in pSym[p]:
        qp = q0 * q                     # crystal-frame pseudo-symmetry applied first
        eu = qu2zyz(qp)
        r = {corr: xc[p]->refinePeak(gln, eu), phase: p, iq: iq}
        insert into the list
for i in [0,n):
    res[i].qu = zyz2qu(res[i].qu)
    res[i].qu = quNp * res[i].qu        # undo the detector-frame rotation (quNp == identity today)
    conjugate res[i].qu                 # crystal->sample  ==>  sample->crystal
```

`refineImage(pat, res)` (`:278-305`): conjugate → left-multiply by `conj(quNp)` → `qu2zyz` → `refine` → `zyz2qu` → left-multiply by `quNp` → conjugate; `res.iq` updated.

`BatchEstimate(bw, nt, np)` (`:189-205`):
```
scl = bw^3 * ln(bw^3) ;  k = 1e-8 ;  tPat = scl*k ;  pps = 1/tPat
batch = max(1, int(pps * 0.61803398874989484820458683436564))   # ~1/phi seconds of work
if ceil(np/batch) < nt*nt:  batch = ceil(np/(nt*nt))
```

ZYZ ↔ quaternion (`xtal/rotations.hpp:973-1020`):
```
zyz2qu: c = cos(b/2); s = sin(b/2); sigma = (g+a)/2 ; delta = (g-a)/2
        qu = [c*cos(sigma), -pijk*s*sin(delta), -pijk*s*cos(delta), -pijk*c*sin(sigma)]
        negate if w<0; if |w| <= rEps then w=0 and orientAxis(qu+1)
qu2zyz: standard atan2 form with the chi <= thr degenerate branches; all angles wrapped to [0, 2pi)
zyz2eu (to Bunge ZXZ): phi1 = a - pi/2 ; Phi = b ; phi2 = g + pi/2
eu2zyz:                a = phi1 + pi/2 ; b = Phi ; g = phi2 - pi/2
```

### 6.3 EBSD driver — `include/modality/ebsd/idx.hpp`

`IndexingData<Real>::initialize(nml, pIpf)` (`:175-311`), in order:
1. `nml.writeFileHeader()` (creates the output HDF5 up front).
2. For each `masterfile`: `MasterSpectra<Real>(file)` (reads `.spx`), then `.resize(nml.bw)`.
3. All phases must share `sig` and `kV` (else throw).
4. Pseudo-symmetry file: single phase only.
5. `pat = PatternFile::Read(patFile, patName, patDims[0], patDims[1])`; sanity-check count and size.
6. `idxMask = nml.roi.buildMask(scanDims)`; if empty, index everything; if `nml.refine`, promote `1 -> 3` (bit 0x01 = index, 0x02 = refine).
7. Geometry: `sampleTilt(phases[0].getSig())`, `cameraTilt(thetac)`, `cameraSize(patDims, delta)`, pattern centre by vendor, `maskPattern(circRad == 0)`, `flipPattern(pat->flipY())`.
8. `threadCount = nThread ? nThread : ThreadPool::Concurrency()`; `batchSize = batchSize ? : Indexer::BatchEstimate(bw, threadCount, numIdx)`.
9. Build orientation maps: `phases.size() + extraScans` copies, where `extraScans = pseudoSym().size()` for a single phase else `1`. Calibration written back as EDAX `x*, y*, z*`.
10. `PatternProcessor::setSize(geom.w, geom.h, circRad, gausBckg, nRegions)`.
11. `gridDim = bw + (bw%2==0 ? 3 : 2)`; `BackProjector(geom, gridDim, sqrt(2), quNp)`.
12. Correlators:
    - **normed**: build `sht(gridDim, bw, Legendre)`; `win = ones(pat->numPix())`; `prj->unproject(win, sph)` (returns 0 and writes 1 at every covered point → binary mask); `sht.analyze(sph, mlm)`. Then for each phase: `sht.synthesize(p.data(), sph)`; square `sph` elementwise; `sht.analyze(sph, flm2)`; `NormalizedCorrelator(bw, p.data(), flm2, p.mirror(), p.nFold(), mlm)`.
    - **unnormed**: `UnNormalizedCorrelator(bw, flm_copy, p.mirror(), p.nFold())`.
13. One `Indexer` per thread (`clone()` of the first).
14. `patBufs[thread] = char[batchSize * pat->imBytes()]`; `ipfBuf = char[scanW*scanH*3]`.
15. `workItem = ebsdWorkItem<PixType, Real>(...)` chosen by `pat->pixelType()`.

`ebsdWorkItem` (`:382-456`) — one call = one batch, `id` is the worker index:
```
indices = pat->extract(buf[id], cnt)                 # mutex-protected pull of `cnt` patterns
for i in indices:
    if msk[i] & 0x01:                                # index
        ref = (msk[i] & 0x02)
        try: idx[id]->indexImage(ptr, res, res.size(), ref)
             for j: om[j].{qu, metric=corr, imQual=iq, phase}[i] = res[j].*
             ipf[3i..3i+3] = pointgroup ipfColor(rotateVector(res[0].qu, {0,0,1}))
        catch: qu = identity, metric = 0, imQual = 0, phase = -1, ipf = 0
    elif msk[i] & 0x02:                              # refine only
        res[0].phase = om[0].phase[i] ; res[0].qu = om[0].qu[i]
        idx[id]->refineImage(ptr, res[0]) ; write back qu/metric/imQual
    ctr++                                            # atomic<uint64_t>
    ptr += bytes
```
Driver (`programs/index_ebsd.cpp:145-175`): `ThreadPool pool(threadCount)`; `batches = ceil(numPat/batchSize)`; schedule `workItem` once per batch; poll `pool.waitAll(1s)` printing progress from the atomic counter; then `idxData.save(...)`.

`ThreadPool` (`include/util/threadpool.hpp`): mutex/condvar task queue; the work function signature is `void(size_t threadId)`; `Concurrency() = max(1, hardware_concurrency())`; destructor waits for all queued tasks.

`ThreadedIqCalc` (`:143-169`, `:463-509`): separate pool (default `Concurrency()/2`) computing `image::ImageQualityCalc::compute` for every pattern, batch size 10.

### 6.4 Namelist — `include/modality/ebsd/nml.hpp`

Fields and defaults (`Namelist::defaults()`, `:186-218`):

| key | field | default | meaning |
|---|---|---|---|
| `ipath` | `ipath` | `""` | input path prefix |
| `patfile` | `patFile` | `"scan.h5"` | pattern file (up1/up2/ebsp/h5/data) |
| `patdset` | `patName` | `"Scan 1/EBSD/Data/Pattern"` | HDF5 dataset path |
| `masterfile` | `masterFiles` | `{"master.h5"}` | one `.spx` per phase |
| `psymfile` | `pSymFile` | `""` | quaternion angle file of pseudo-symmetries (single phase only) |
| `patdims` | `patDims[2]` | `640, 480` | binned pattern w, h |
| `circmask` | `circRad` | `-1` | `-1` none, `0` largest inscribed circle, `>0` radius in px |
| `gausbckg` | `gausBckg` | `false` | 2-D Gaussian background subtraction |
| `nregions` | `nRegions` | `10` | AHE tiles per axis (`0` = no AHE) |
| `delta` | `delta` | `50.0` | pixel size on the scintillator, microns |
| `vendor` | `ven` | `"EMsoft"` | pattern-centre convention |
| `pctr` | `pctr[3]` | `0, 0, 15000` | pattern centre (interpretation per vendor) |
| `thetac` | `thetac` | `10.0` | camera tilt, degrees (positive below horizontal) |
| `scandims` | `scanDims[2]`,`scanSteps[2]` | `256,256,1.0,1.0` | scan size + step, or a scan-file name |
| `scanname` | `scanName` | `""` | HDF5 scan group |
| `roimask` | `roi` | none | ROI: `0`, `x0,y0,dx,dy`, or a mask file |
| `bw` | `bw` | `68` | SH bandwidth; `2*bw-1` should be a product of small primes |
| `normed` | `normed` | `true` | normalized vs unnormalized cross correlation |
| `refine` | `refine` | `true` | Newton refinement vs tri-quadratic interpolation only |
| `nthread` | `nThread` | `0` | `0` = auto |
| `batchsize` | `batchSize` | `0` | `0` = `BatchEstimate` |
| `opath` | `opath` | `""` | output path |
| `datafile` | `dataFile` | `"SphInx_Scan.h5"` | HDF5 output |
| `vendorfile` | `vendorFile` | `"reindexed.ang"` | ang/ctf output (optional) |
| `ipfmap` | `ipfName` | `"ipf.png"` | IPF map (optional) |
| `qualmap` | `qualName` | `"qual.png"` | XC map (optional) |

Suggested `bw` values (`:415-416`): `53, 63, 68, 74, 88, 95, 113, 122, 123, 158, 172, 188, 203, 221, 263, 284, 313`; parameter sweep `53, 68, 88, 113, 158, 203, 263, 338`.

**Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D1, D9)**: the `circmask` default `-1` means `IndexEBSD` applies **no detector circle** (`Geometry::maskPattern(circRad == 0)` is false) and no histogram mask (`setSize(w, h, r = -1, ...)` builds the all-ones `BckgSub2D`) by default; the port's `SphericalBackProjector(circular_mask=False)` and `_preprocess_pattern(good_pixels=None, n_regions=10, gaussian_background=False)` defaults follow it, so Phase 10 parity runs against `IndexEBSD` defaults need no keyword. `circmask = 0` -> `circular_mask=True` + `good_pixels=_circular_mask(shape)`; `circmask = r > 0` -> `good_pixels=_circular_mask(shape, r)` and `circular_mask=False` (section 8 item 21).

`sanityCheck()` (`:621-639`):
```
patDims in [2, 16384]
circRad >= -1
nRegions in [0, min(patDims)]
detector width = delta*patDims[0]/1000 must be in [5, 90] mm
thetac in [-60, 60] deg
scanDims >= 1
bw in [16, 512]
nThread >= 0 ; batchSize >= 0
patFile, masterFiles, dataFile non-empty
```

---

## 7. Master pattern handling — `include/idx/master.hpp`

### 7.1 `MasterData<Real>` (`:47-85`)
Holds `pSm` (pseudo-symmetric quaternions), `phs` (`xtal::Phase` = point group + lattice), `sig` (sample tilt, deg), `kv` (kV).
`addPseudoSym(fn)` (`:223-233`): reads an EMsoft quaternion angle file, skipping the identity.

### 7.2 `MasterPattern<Real>` (`:89-149`)
`nh`, `sh` (each `dim*dim`), `dim`, `lyt` (`Lambert` or `Legendre`).

`read(fileName)` (`:242-347`) — EMsoft HDF5:
```
sig  = /NMLparameters/MCCLNameList/sig          # sample tilt
kv   = /NMLparameters/MCCLNameList/EkeV
phs.readMaster(/CrystalData)
accum_e = EMData/MCOpenCL/accum_e               # (x, y, E) Monte-Carlo counts
weights[e] = sum over (x,y) of accum_e ; normalized to sum 1
detect EBSD vs ECP by the presence of EMData/EBSDmaster or EMData/ECPmaster
mLPNH / mLPSH  (EBSD: {atom, energy, x, y};  ECP: {atom, x, y} with a single unit weight)
sum over atoms, then energy-weighted average -> nh, sh (float32 -> Real)
dim = dims[2] ;  lyt = Lambert
```

`resize(nDm)` (`:354-374`): Lambert only; DCT-based `image::Rescaler`, with the explicit FFTW correction `*= 0.5/nhScaled.size()`.

`toLegendre(nDm)` (`:381-416`) — **this is how a square-Lambert master becomes a square-Legendre master**:
```
dimScaled = round(sqrt(2)*nDm) ; resize(dimScaled)          # oversample first for better interpolation
xyz = square::legendre::normals(nDm)                        # target grid directions (north hemi)
for each grid point i:
    (X,Y) = lambert::sphereToSquare(xyz[i])                 # note: uses |z| -> same square for both hemis
    p.bilinearCoeff(X, Y, dimScaled, dimScaled)
    lgNh[i] = p.interpolate(nh) ;  lgSh[i] = p.interpolate(sh)
lyt = Legendre ; dim = nDm
```

`toLambert(nDm)` (`:423-473`): reverse, using `legendre::boundingInds` + **nearest neighbour** among the 4 bounding points (chosen by max dot product); the code notes barycentric would be better.

Symmetrisation helpers (all operate on the *real-space* grid, do **not** update the crystal structure):
- `makeNFold(n, m)` (`:483-516`): for every ring, read the ring, optionally impose an in-ring mirror (`m == 1`: mirror at `phi = i*180/n`, reverse-copy the first half of the repeat; `m == 2`: mirror at `phi = i*180/n + 90/n`, reverse-copy the first and third quarters), then replicate the first `1/n` of the ring `n` times at `round(repeatNum*j)`. Exact only for `n in {2,4}`; approximate otherwise (`:135`).
- `makeZMir()` (`:141`): `sh = nh`.
- `makeInvSym()` (`:521-527`): `sh[(dim-1-j)*dim + (dim-1-i)] = nh[j*dim+i]`.
- `matchEquator()` (`:531-538`): copies the 4 border rows/columns from `nh` to `sh`.
- `canRescale()` = `lyt == Lambert`.

### 7.3 `MasterSpectra<Real>` (`:153-206`, `:550-640`)
Storage: `alm` with `a^l_m` at `alm[bw*m + l]` — **m-major**, `mBw x mBw`.
```
nFold()  = pointGroup().zRot()       # 1,2,3,4,6      -> 'fn' / fNf
mirror() = pointGroup().zMirror()    # bool           -> 'fm' / fMr
invSym() = pointGroup().inversion()
removeDC(): alm[0] = 0
```

`MasterSpectra(mp, bw, nrm = true)` (`:550-595`):
```
dimLg = bw + 2 + (bw%2==0 ? 1 : 0)                   # odd
if mp.lyt != Legendre: mp.toLegendre(dimLg)
dim = mp.dim
if nrm:
    if dim < dimLg: throw "insufficient grid resolution for requested bandwidth"
    omega   = square::solidAngles(dim, Legendre)
    weights[i] = omega[ringNum(dim, i)]
    halve weights on the 4 border rows/columns (equator double counting)
    totW = sum(weights)
    mean  = ( <weights, nh> + <weights, sh> ) / totW           # <-- see gotcha 8.7
    nh -= mean ; sh -= mean
    stdev = sqrt( ( <weights, nh^2> + <weights, sh^2> ) / (2*totW) )
    nh /= stdev ; sh /= stdev
if dim % 2 == 0: throw
phs, kv, sig copied from mp ; mBw = bw
alm.resize(bw*bw)
DiscreteSHT<Real>::Legendre(dim).analyze(nh, sh, alm, bw, bw)
# note: the line `alm[0] = 0` is present but commented out (:594)
```

`resize(bw)` (`:601-614`): zero-pad up (`m` descending) or crop down (`m` from 1) in the `m`-major packing.

`read(fileName)` (`:619-640`) — EMSphInx `.spx` binary:
```
kv  = header.beamEnergy() ; sig = header.primaryAngle()
mBw = harmonics.bw()
phs.pg = PointGroup(mpData.sgEff())          # effective space group -> point group
lattice parameters if numXtal()==1
alm.resize(mBw*mBw, 0) ; harmonics.unpackHarm(harmonics.alm.data(), alm.data())
```
`UnpackHarm(in, out, bw, n, f)` (`build/_deps/shtfile-src/sht_file.in.hpp:1766-1830`) — compression flags:
```
0x01 inv  : keep only l % 2 == 0
0x02 mirZ : keep only (l+m) % 2 == 0
0x04 mirY : each row is strictly REAL         (Nmm-type group)
0x08 mirX : rows with m % (2n) == 0 real, others strictly IMAGINARY (rotated Nmm)
(0x04 and 0x08 are mutually exclusive)
rows with m % n != 0 are all zero ;  entries with l < m are zero padding
```

Generating a `.spx`: `programs/mp2sht.cpp` uses `bw = 384`, `nrm = true`, `iprm[10] = 1` (latitude grid type = Legendre).

---

## 8. Numerical constants, tolerances, and gotchas

1. **`dim` must be odd** everywhere (`square_sht.hpp:339`, `master.hpp:587`, `legendre::normals:825`, `makeNFold:484`). Even side lengths are partially coded but unsupported.
2. **Bandwidth ↔ grid size:** `dim = bw + (bw%2==0 ? 3 : 2)`, equivalently `bw <= dim - 2` for Legendre and `bw <= (dim-1)/2` for Lambert. `MasterSpectra` uses `dimLg = bw + 2 + (bw even ? 1 : 0)` — identical.
3. **`bw` is exclusive**: `l` runs `0 .. bw-1`. The namelist restricts `bw` to `[16, 512]`.
4. **`slP = fastSize(2*bw-1)`** — zero padding to the next `{2,3,5,7,11,13}`-smooth size. `bwP = slP/2 + 1`. Choose `bw` so `2*bw-1` is already smooth if you want no padding.
5. **FFT normalisation:** FFTW `r2c`/`c2r` and `REDFT10`/`REDFT01` are unnormalised. In `DiscreteSHT` the `1/N_phi(y)` is folded into `w_y` (`w_y = 4*pi * what_y / max(1,8y)`); in the 3-D correlation nothing is normalised, so a NumPy `irfftn` must be multiplied by `slP^3`.
6. **Condon–Shortley:** the Schaeffer recursion omits `(-1)^m`, so odd `m` are negated in `analyze` (`:439`) and `synthesize` (`:554-555`). Forget this and every odd-`m` coefficient flips sign.
7. **`MasterSpectra` mean is 2x too large** (`master.hpp:572-573`): the two hemisphere inner products are summed but divided by `totW` (one hemisphere's weight sum) rather than `2*totW`. The stdev at `:581` *does* use `2*totW`. Net effect: `a^0_0` is not zero after normalisation. `removeDC()` exists and the explicit `alm[0] = 0` is commented out at `:594`.
8. **Corner weights are quartered** in `MasterSpectra` (`:565-568`): the four loops that halve the border rows/columns overlap at the corners, so corner pixels get `/4` instead of `/2`.
9. **`BackProjector` south-hemisphere index bug** (`detector.hpp:552`): `p.idx = i` rather than `dim*dim + i`, so southern back-projection would collide with the northern grid and the south half of `sph` would never be written. **It is latent** because `interpolatePixel` rejects `n[2] < 0` (`:336`) and standard EBSD/ECP geometry puts the entire footprint in the northern hemisphere. If you enable `northPoleQuat()` or a geometry with `alpha` large enough that part of the detector sees `z < 0`, this becomes a real bug.
    **Addendum (2026-08-17, `specs/2026-08-17-spherical-back-projection/requirements.md` D4)**: the south loop is **unreachable** as written -- `interpolatePixel` rejects `n[2] < 0` at `:336` before any geometry test, so no geometry reaches it without also editing `interpolatePixel`. The port gathers the north grid only and returns an all-zero south hemisphere (see the 4.4 addendum).
10. **`northPoleQuat()` returns identity** (`detector.hpp:455-459`); the "rotate the detector footprint to the pole" optimisation is disabled, and `Indexer` still applies `quNp` (a no-op) plus the final conjugation.
11. **`interpPeak` bounds check ignores `x[2]`** (`sht_xcorr.hpp:421`): `max(|x[0]|, max(|x[1]|, |x[0]|))`. A large alpha over-step is not caught.
12. **`extractNeighborhood` glide for even `slP`** (`sht_xcorr.hpp:530-531`): `m < bwP ? m + bwP - 1 : m - bwP`. For odd `slP` both branches are `m + (slP-1)/2 (mod slP)` — consistent. For even `slP` the second branch is off by one relative to `+slP/2`. `eulerIndex` (`:559-563`) uses exact `fmod(x + slP/2, slP)`.
    **Addendum (2026-08-17, `specs/2026-08-17-spherical-cross-correlation/requirements.md` D5, D6)**: this item describes only the even-`slP` off-by-one. Phase 4 found that the shift is applied per slot index rather than per glided plane, that the gamma slot `:531` receives the identical shift, and that at `k0 = bwP - 1` the even-`slP` reads run past the end of `xc` (see the D5 erratum in section 3.4); `eulerIndex` (no caller) has no `beta` wrap and rounds `beta = 0` to `k = bwP` for odd `slP` (D6 erratum in section 3.4).
13. **`n` only loops to `bwP` in `compute`** (`:756`); for even `slP` the row `n = slP/2` is written twice (`v` then `vnc`), but that row is always in the systemic-zero branch because `bwP - 1 >= mBw` in that case, so it is harmless.
14. **`interpolateMaxima`** silently falls back to `x = 0` if Newton does not converge in 25 iterations (`:1350`); tolerance `sqrt(machine eps)`.
15. **`refinePeak` tolerances:** `absEps = eps*2*pi/slP` with `eps = 0.01` by default; `maxIter = 15`; `euEps = sqrt(machine eps)` for the `beta ≈ 0, +/-pi` degeneracy; monotone step-shrinking is enforced; Cholesky failure is *used deliberately* to reject non-positive-definite (saddle) Hessians. On any failure the original interpolated orientation is returned with the correlation re-evaluated.
16. **`computeWeightsSkip` precision guard** (`:1058`): throws `"insufficient precision to accurately compute ring wieghts"` if `sum(w) - 1 > cbrt(machine eps)/64`. This is the practical bandwidth ceiling for the Lambert grid in double precision; the weight solve is `O(dim^4)` and is the dominant construction cost.
17. **`w_jkm` integer overflow** (`wigner.hpp:227`): 32-bit ints cap `k` at ~215; 64-bit at ~55000. Use `int64` / Python ints and cast late.
18. **Wigner tables contain garbage for `j < max(k,m)`** (`wigner.hpp:108`, `:124`) — never read those slots; the correlation loops always start at `j = max(m, k, n)`.
19. **`d(...)` returns `NAN` when `j < max(|k|,|m|)`**.
20. **Symmetry flags:** `fMr` = `pointGroup().zMirror()` implies `flm[m*bw+j] == 0` unless `(m+j) % 2 == 0`; `fNf` = `pointGroup().zRot()` (∈ {1,2,3,4,6}) implies `flm[m*bw+j] == 0` unless `m % fNf == 0`. The pattern side is always assumed to have `glnFold = 1`, `gMir = false`. When `fMir` is set, the `j` loop steps by 2 (`dJ = 2`) after adjusting `start` for parity.
21. **`circmask` inconsistency** (`modality/ebsd/idx.hpp:230`): the *detector* circular mask is enabled only when `circRad == 0` (`geom.maskPattern(nml.circRad == 0)`), while the *background-subtraction/AHE* mask uses `circRad` fully (`-1` none, `0` auto, `>0` explicit radius). So `circmask = 25` masks the histogram but not the back projection.
22. **Gaussian fit off-by-one** (`gaussian.hpp:273, 280, 308-315`): the fit is performed on `cWrk[1 .. w-2]` with implicit abscissa `0..w-3`, but the evaluated background uses the raw column index `i`. The fitted mean is therefore shifted by one pixel.
23. **`sph` is not cleared between patterns** — safe only because the interpolation LUT is fixed; a port must zero it once at allocation.
24. **`Indexer::sum2` is dead** — the confidence is not divided by the pattern norm (see §3.10).
25. **`Geometry::rescale(Real scale)`** passes `wNew` twice (`detector.hpp:425`); use the two-argument overload.
26. **`flip` is applied after the frame bounds test** in `interpolatePixel` (`:359-360`), which is correct only because the test is symmetric in `Y`.
27. **`alpha = (90 - sTlt + dTlt) deg`** is the single angle that encodes both tilts; `|alpha| > 90 deg` throws. `dOmg`/`sOmg` != 0 throws `"omega tilt not yet supported"`.
28. **`fastSize`** (`util/fft.hpp:438-491`) returns `max(1,x)` for `x <= 16`, else the smallest `{2,3,5,7,11,13}`-smooth number `>= x`.
29. `emsphinx::Constants<Real>::pi/pi2/pi_2` are given to 72 decimals; `pi2 = 2*pi` (not `pi/2` — `pi_2` is `pi/2`).
30. `MasterPattern::toLegendre` oversamples the Lambert grid by `round(sqrt(2)*nDm)` first; `BackProjector` oversamples the detector by `sqrt(2)` (`fct` argument, `modality/ebsd/idx.hpp:259`) beyond `scaleFactor(dim)`.
31. **`bilinearCoeff` pixel stretch** (`util/image.hpp:528`; Phase 5, `specs/2026-08-17-spherical-back-projection/requirements.md` D2): `x = X (w - 1)` treats the pixel centres `0 .. w-1` as spanning the whole physical width, displacing samples by up to `+-0.49` px at the edges. Not reproduced by the port (kikuchipy pixel-centre convention + DCT sampling convention; the coarse lock cannot tell the two apart). Related: EMSphInx' `patternCenterBruker` equates `xpc = (x*_B - 0.5) w`, kikuchipy's pre-v5 EMsoft relation -- use `vendor = 'Bruker'` with kikuchipy's `pc` in `IndexEBSD` namelists (section 4.3 addendum).
32. **`solidAngle` divisor** (`detector.hpp:401-415`; Phase 5, D3): the loop evaluates `(gridRes + 1)^2` northern points, the divisor `gridRes^2 + (gridRes - 2)^2 = 500002` is the count for `gridRes - 1`; the consistent count is `502004`, the fraction is 0.4 % high. Ported literally for `Rescaler` size parity, labelled a quirk. Also: `omgS == 2 dim^2 - 4 (dim - 1)` exactly; the C++ applies the circle twice with different centres/radii (unrescaled in `solidAngle`, rescaled in the LUT loop -- an ellipse in detector pixels for rectangular detectors; the port uses the physical circle once); the two-argument `rescale` has no empty-geometry guard (the port raises `ValueError` for `rescaled_shape < 1 px` and `n_points == 0`) (section 4.4 addendum).
33. **pocketfft's DCT of a constant is inexact** (a porting note, Phase 5, D5): AC terms `~1e-11`, so the `stdev == 0` mask path of `unproject` is not taken for a constant pattern under `scipy.fft`; the port builds the window mask directly and short-cuts `ptp == 0` before the DCT (`iq` `1.0` non-zero constant / `0.0` all-zero) (section 4.5 addendum).
34. **`Model::fit` non-convergence on an exact integer-mean Gaussian** (`gaussian.hpp:225-228`; Phase 5, D9): `ss = 0` -> metric `0/0 = NaN` -> flat fallback, reproduced (`error_model="numpy"`); `rWrk`/`cWrk` initialised to `im[0]` even when masked; `solve::cholesky` comparisons are false for NaN, so a flat input runs 50 NaN iterations and reports non-convergence (comparisons ported in the C++ direction; status platform-dependent, recorded) (section 5.2 addendum).
35. **AHE and IQ behaviours** (`ahe.hpp`, `image.hpp:489-507`; Phase 5, D5, D9): masked-out pixels are equalised (the mask only shapes the histograms), a uniform image maps to 255, the mosaic AHE equals `skimage` CLAHE (`kernel_size = shape // n_regions`, `clip_limit=0`, `nbins=256`) to 6 gray levels for dividing tiles and diverges otherwise; `imageQuality` is offset-dependent (DC in `sumP`) and correlates 0.62 with kikuchipy's `get_image_quality` (sections 5.3, 5.4 addenda).
36. **`circmask` default `-1`** (`nml.hpp:186-218`; Phase 5, D1, D9): `IndexEBSD` applies no detector circle and no histogram mask by default; the port's `circular_mask=False` / `good_pixels=None` defaults follow it, and `circmask = 0` / `r > 0` map to `circular_mask=True` + circle mask / circle mask only (item 21; section 6.4 addendum).

---

## 9. Existing tests in `test/sht/` (reference tests for a port)

### 9.1 `test/sht/square_sht.cpp` — `DiscreteSHT` round trip
`testSingleSHT<Real>(bw, layout, maxEps, avgEps, os)` (`:90-148`):
```
seed = 0, std::mt19937_64, uniform_real_distribution<Real>(-1, 1)
refSpec[m*bw + l] for l >= m:  (real, 0)      if m == 0
                               (real, imag)   otherwise    ; everything else 0
dimLam = 2*bw + 1
dimLeg = bw + (bw%2 == 0 ? 3 : 2)
sht = DiscreteSHT(dim, bw, layout)
func = synthesize(refSpec)          # dim*dim*2
newSpec = analyze(func)
fail if max(|Re delta|, |Im delta|) > maxEps for any (m,l), or mean > avgEps
```
`testSHT` (`:154-201`):
```
double: maxEps = 0.005 , avgEps = 0.00005
        Legendre bw = 4..384 ; Lambert bw = 4..128
float (only if EM_USE_F): maxEps = 0.020 , avgEps = 0.00020
        Legendre bw = 4..192 ; Lambert bw = 4..64
```
This is the single most useful port test: **synthesize → analyze must be the identity to ~5e-3 absolute per coefficient**.

### 9.2 `test/sht/sht_xcorr.cpp` — spherical cross correlation
Helpers:
- `randomSphere(gen, dim, mir, nFld)` (`:108-127`): `MasterPattern` on the Legendre layout, `nh` filled `U(-1,1)`; if `mir` then `sh = nh`, else `sh` is independently random and `matchEquator()` is applied; then `makeNFold(nFld)` if `nFld > 1`.
- `randomRotation(gen)` (`:133-136`): `Quat(|U|, U, U, U).normalize()`.
- `randomPair(bw, mir, nFld, flm, gln)` (`:146-175`): seed 0; `dim = bw + (bw%2==0?3:2)`; `MasterSpectra(mp, bw, nrm=false)` → `flm`; **exact** zeroing of rows `m % nFld != 0`; random rotation → `qu2zyz` → `wigner::rotateHarmonics(bw, flm, gln, eu)`; returns the quaternion.

`testCorr(bw, mir, nFld, qu, qr)` (`:184-203`): `Correlator<Real> s2(bw); s2.correlate(flm, gln, mir, nFld, eu, ref=true)`; `zyz2qu(eu)`; returns the misorientation angle in degrees `180*acos(min(dot,1))/pi`. (Both quaternions are conjugated before being reported.)

`testNCorr(bw, mir, nFld, ...)` (`:212-287`): builds an azimuthal-wedge mask (`theta in [-30 deg, +60 deg]`, plus `z < cos(45 deg)` for the south hemisphere), band-limits the random function and the mask by an analyze/synthesize round trip, computes `flm2 = SHT(f^2)`, rotates `flm`, **masks the rotated function in real space**, re-analyzes → `gln`, then `NormalizedCorrelator(bw, flm, flm2, mir, nFld, mlm).correlate(gln, eu, ref=true)`.

`runTests<double>` (`:292-395`):
```
eps  = cbrt(FLT_EPSILON)  ~ 4.9e-3   ("~6e-6 degrees for double, 0.005 degrees for float")
sizes = {53, 68, 88, 113, 123, 158,          # already fast FFT sizes
         54, 55, 56, 57, 58, 59, 60, 62, 64} # need zero padding
1) testCorr(bw, false, 1) for each size            -> delta <= eps
2) testNCorr(bw, false, 1) for each size           -> delta <= eps*10
3) point groups {112, 11m, 112/m, 3, 4, 4/m, 6, 6/m}, bw = 53..63:
   eps = sqrt(eps)*5  (~0.012 deg for double, ~0.35 deg for float)
   testCorr  with (pg.zMirror(), pg.zRot()) ; if delta > eps compare disorientation pg.disoQu(...)
   testNCorr likewise
```
Comment at the top (`:39-41`): "Tests are still missing for normalized cross correlation" (stale — `testNCorr` exists).

### 9.3 `test/sht/wigner.cpp` (137 kB, mostly hard-coded Mathematica reference tables)
`runTests<Real>` (`:875-877`) = `testDjkm && testTables(15) && testDerivatives`, run for **both `float` and `double`** (`:89-90`).

- `testDjkm` (`:112-...`): tables `N[Table[WignerD[{j,k,m}, Pi/2], {j,0,4},{k,-4,4},{m,-4,4}], 32]` and the analogous tables for `beta = pi/3`, `2pi/3` (and their negatives). `Num = 5` (`j in [0,5)`, `k,m in (-5,5)`). Tolerance `eps = 2 * machine eps`. Functions exercised: `d(j,k,m)` (the `pi/2` special case), `d(j,k,m,t,nB)` for `t = 0, 0.5, -0.5` with `nB = false/true`, `dSign`, and `D(j,k,m,eu)`.
- `testTables(bw = 15)` (`:405-...`): `beta = 0.9708055194` (= `pi/2/phi`, deliberately not a symmetry point), `t = cos(beta)`. Compares, element by element against `d(j,k,m,t,nB)`:
  - `dTable(bw, t, false, table)` and `dTable(bw, t, true, table)`;
  - `dTablePreBuild(bw, pE, pW, pB)` then `dTablePre(bw, t, false/true, table, pE, pW, pB)`;
  - `dTable(bw, table, trans=false)` and `dTable(bw, table, trans=true)` against `d(j,k,m,0,false)`.
- `testDerivatives` (`:563-...`): Mathematica `D[WignerD[{j,k,m},beta],{beta,1}]` and `{beta,2}` tables at `beta = pi/3` and `2pi/3` (i.e. `t = +/-0.5`) plus the negative-`beta` variants. Tolerance `eps = 24 * machine eps`.
- **No unit test exists for `rotateHarmonics`** (`:874`) — but `sht_xcorr.cpp` exercises it indirectly.

### 9.4 Reference values you can reproduce for a port
- `analyze` of `f == 1` on any grid gives `a^0_0 = sqrt(4*pi)`, all other coefficients 0.
- `sum_y what_y == 1` for `computeWeightsSkip` before scaling.
- `w0 == 4*pi` exactly for odd `dim`.
- Round trip `synthesize -> analyze` is the identity to `5e-3` per coefficient up to `bw = 384` (Legendre) in double.
- `Correlator.correlate` recovers a rotation applied via `rotateHarmonics` to `< 1e-5` degrees for `bw >= 53`.

---

## 10. Minimal port checklist (order of implementation)

1. `squareToSphere` / `sphereToSquare`; `lambert::cosLats`; `legendre::roots` (or `numpy.polynomial.legendre.leggauss`); `legendre::normals`; `ringNum`; `solidAngles` (ring version).
2. Ring index tables (`readRing`/`writeRing` as precomputed index arrays).
3. `computeWeightsSkip` (dense `Nt-1` Chebyshev matrix + `np.linalg.solve`, then the `4*pi/N_phi` scaling).
4. `amn`/`bmn` tables + the `analyze`/`synthesize` ring loops (Numba-friendly: the inner `j` recursion is scalar and sequential).
5. `wigner::dTable` at `pi/2` (`trans=true` layout) and `dTablePre`/`dTablePreBuild` for arbitrary beta.
6. `Correlator::compute` (the `k, n, m, j` loop) + `irfftn * slP^3`, then `findPeak`, `extractNeighborhood`, `interpolateMaxima`, `interpPeak`.
7. `derivatives` + `refinePeak` (Newton with Cholesky and the degeneracy fallbacks).
8. `NormalizedCorrelator` constants (`rDen`) and `denominator`.
9. Detector `interpolatePixel` + `BiPix` LUT + DCT rescaler + `unproject`.
10. `PatternProcessor` (Gaussian background, AHE), `Rescaler`, `imageQuality`.
11. `MasterSpectra` (`toLegendre` + weighted normalisation + `analyze`), `.spx` `UnpackHarm`.
12. `Indexer.indexImage` glue + ZYZ↔quaternion conversions.
