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

"""The device-only CUDA kernels of the HREBSD IC-GN GPU backend: the
cupy :class:`~kikuchipy.indexing._hrebsd._batched.KernelNamespace`.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Requirements D21.4 (the two device precisions), D21.6 (the batched
semantics), D21.7 (determinism) and D21.9.5 (the kernel interface)
govern.  The numpy twin of the ``"float64"`` build lives in
``_batched.py`` and fixes the algebra these kernels follow term for
term (the order of the per-pattern sums in its module docstring); the
gated per-kernel A/B oracle of V9(f) compares the two on identical
inputs.

Determinism (D21.7.2): no atomic operation anywhere.  Every reduction
is two-stage and fixed-order: a fixed shared-memory tree per block,
then a sequential pass over the block partials of one pattern.  The
launch layout is :func:`~kikuchipy.indexing._hrebsd._batched.\
_launch_layout` of the subregion pixel count only, never of B
(D21.7.3).  Compiled without fast math and with ``--fmad`` at the NVRTC
default (D21.4): the mixed build's B-spline weights multiply by the
reciprocal constant ``1/6`` written in the source, the float64 build's
divide by 6 as the CPU does.

CuPy is imported inside :func:`make_cupy_kernel_namespace` only (after
the gate), never at module scope (D21.13); every kernel is CONSTRUCTED
through the attribute ``cupy.RawKernel`` on every call, with no
``cupy.RawModule`` and no Python-level cache across calls.
"""

import math

import numpy as np

from kikuchipy.indexing._hrebsd import _batched

# NVRTC options of every kernel, the spline prefilter included: the
# C++ dialect only, so ``--fmad`` stays at the NVRTC default and fast
# math is never on (D21.4)
_KERNEL_OPTIONS: tuple[str, ...] = ("--std=c++14",)

# The per-precision preambles: the pixel type of the per-pixel
# arithmetic and the per-thread partials (D21.4)
_PREAMBLE = {
    "mixed": "#define HREBSD_MIXED 1\ntypedef float pixel_t;\n",
    "float64": "#define HREBSD_MIXED 0\ntypedef double pixel_t;\n",
}

_SOURCE = r"""
#define N_SUMS 10
#define HREBSD_NAN __longlong_as_double(0x7ff8000000000000LL)

// Fold a coordinate into [0, n - 1] through the whole-sample mirror
// boundary IN FLOATING POINT, the CPU's order (D21.6.3)
__device__ __forceinline__ double fold_coordinate(double c, int n) {
    const double period = (double)(2 * n - 2);
    if (c < 0.0) c = -c;
    c = fmod(c, period);
    if (c > (double)(n - 1)) c = period - c;
    return c;
}

// Fold a support index of the basis, within one period of the frame
__device__ __forceinline__ int fold_index(int k, int n) {
    const int period = 2 * n - 2;
    if (k < 0) k = -k;
    k = k % period;
    if (k > n - 1) k = period - k;
    return k;
}

// The cubic B-spline weights.  The mixed build multiplies by the
// reciprocal constant 1/6 in place of the eight f32 divisions (D21.4);
// the float64 build, the parity and debug mode, divides by 6 as the
// CPU's ``_bicubic_evaluate`` does
template <typename T>
__device__ __forceinline__ void basis(T t, T *w) {
    const T one_minus = (T)1 - t;
    const T t2 = t * t;
    const T t3 = t2 * t;
#if HREBSD_MIXED
    const T sixth = (T)(1.0 / 6.0);
    w[0] = one_minus * one_minus * one_minus * sixth;
    w[1] = ((T)3 * t3 - (T)6 * t2 + (T)4) * sixth;
    w[2] = ((T)-3 * t3 + (T)3 * t2 + (T)3 * t + (T)1) * sixth;
    w[3] = t3 * sixth;
#else
    w[0] = one_minus * one_minus * one_minus / (T)6;
    w[1] = ((T)3 * t3 - (T)6 * t2 + (T)4) / (T)6;
    w[2] = ((T)-3 * t3 + (T)3 * t2 + (T)3 * t + (T)1) / (T)6;
    w[3] = t3 / (T)6;
#endif
}

// The cubic B-spline at the cell (ix, iy) with fractions (tx, ty)
template <typename T>
__device__ __forceinline__ T spline_value(const float *__restrict__ c, int H,
                                          int W, int ix, int iy, T tx, T ty) {
    T wx[4], wy[4];
    basis<T>(tx, wx);
    basis<T>(ty, wy);
    int cols[4], rows[4];
    if (ix >= 1 && ix + 2 <= W - 1) {
        cols[0] = ix - 1; cols[1] = ix; cols[2] = ix + 1; cols[3] = ix + 2;
    } else {
        for (int k = 0; k < 4; k++) cols[k] = fold_index(ix - 1 + k, W);
    }
    if (iy >= 1 && iy + 2 <= H - 1) {
        rows[0] = iy - 1; rows[1] = iy; rows[2] = iy + 1; rows[3] = iy + 2;
    } else {
        for (int k = 0; k < 4; k++) rows[k] = fold_index(iy - 1 + k, H);
    }
    T value = (T)0;
    for (int a = 0; a < 4; a++) {
        const float *r = c + (size_t)rows[a] * W;
        const T inner = wx[0] * (T)r[cols[0]] + wx[1] * (T)r[cols[1]] +
                        wx[2] * (T)r[cols[2]] + wx[3] * (T)r[cols[3]];
        value += wy[a] * inner;
    }
    return value;
}

// The cell and fraction of a FINITE array-frame coordinate: folded
// in double, range guarded, before any integer conversion
__device__ __forceinline__ void cell_of(double c, int n, int &i, double &t) {
    if (!(c >= 0.0 && c <= (double)(n - 1))) c = fold_coordinate(c, n);
    const double f = floor(c);
    i = (int)f;
    t = c - f;
}

// Batched gather (D21.6.3): grid (blocks per pattern, B), one value
// per subregion pixel, a non-finite coordinate flagged before the fold
extern "C" __global__ void hrebsd_gather(
    const float *__restrict__ coefficients, const int H, const int W,
    const pixel_t *__restrict__ xi_x, const pixel_t *__restrict__ xi_y,
    const int *__restrict__ columns, const int *__restrict__ rows,
    const int n_pixels, const double *__restrict__ matrices,
    const double offset_x, const double offset_y,
    pixel_t *__restrict__ values, int *__restrict__ bad_partials) {
    const int b = blockIdx.y;
    const double *m = matrices + 9 * (size_t)b;
    const float *c = coefficients + (size_t)b * H * W;
    pixel_t *out = values + (size_t)b * n_pixels;
#if HREBSD_MIXED
    // The displacement form in f32 on the renormalised carried matrix
    const float h0 = (float)(m[0] - 1.0), h1 = (float)m[1], h2 = (float)m[2];
    const float h3 = (float)m[3], h4 = (float)(m[4] - 1.0), h5 = (float)m[5];
    const float h6 = (float)m[6], h7 = (float)m[7];
#endif
    int bad = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pixels;
         i += gridDim.x * blockDim.x) {
#if HREBSD_MIXED
        const float x = xi_x[i], y = xi_y[i];
        const float s1 = h6 * x + h7 * y;
        const float inv = 1.0f / (1.0f + s1);
        const float ux = (h0 * x + h1 * y + h2 - x * s1) * inv;
        const float uy = (h3 * x + h4 * y + h5 - y * s1) * inv;
        if (!isfinite(ux) || !isfinite(uy)) {
            bad = 1;
            out[i] = (pixel_t)HREBSD_NAN;
            continue;
        }
        int ix, iy;
        float tx, ty;
        const float fx = floorf(ux), fy = floorf(uy);
        const bool near = fabsf(ux) < 1.0e6f && fabsf(uy) < 1.0e6f;
        ix = near ? columns[i] + (int)fx : -1;
        iy = near ? rows[i] + (int)fy : -1;
        tx = ux - fx;
        ty = uy - fy;
        if (!(ix >= 0 && (ix < W - 1 || (ix == W - 1 && tx == 0.0f)))) {
            double t;
            cell_of((double)columns[i] + (double)ux, W, ix, t);
            tx = (float)t;
        }
        if (!(iy >= 0 && (iy < H - 1 || (iy == H - 1 && ty == 0.0f)))) {
            double t;
            cell_of((double)rows[i] + (double)uy, H, iy, t);
            ty = (float)t;
        }
        out[i] = spline_value<float>(c, H, W, ix, iy, tx, ty);
#else
        const double x = xi_x[i], y = xi_y[i];
        const double scale = m[6] * x + m[7] * y + m[8];
        const double X = (m[0] * x + m[1] * y + m[2]) / scale + offset_x;
        const double Y = (m[3] * x + m[4] * y + m[5]) / scale + offset_y;
        if (!isfinite(X) || !isfinite(Y)) {
            bad = 1;
            out[i] = (pixel_t)HREBSD_NAN;
            continue;
        }
        int ix, iy;
        double tx, ty;
        cell_of(X, W, ix, tx);
        cell_of(Y, H, iy, ty);
        out[i] = spline_value<double>(c, H, W, ix, iy, tx, ty);
#endif
    }
    // A block-wide OR, no atomic: one flag per block
    const int any_bad = __syncthreads_or(bad);
    if (threadIdx.x == 0)
        bad_partials[(size_t)b * gridDim.x + blockIdx.x] = any_bad ? 1 : 0;
}

// The N_SUMS per-pattern sums of the shifted-data form (D21.4,
// D21.6.5): per-thread partials in pixel_t over a fixed run of
// pixels, a fixed f64 shared-memory tree per block, one partial row
// per block
extern "C" __global__ void hrebsd_pixel_sums(
    const pixel_t *__restrict__ values, const double *__restrict__ shifts,
    const pixel_t *__restrict__ xi_x, const pixel_t *__restrict__ xi_y,
    const pixel_t *__restrict__ gradient_x, const pixel_t *__restrict__ gradient_y,
    const int n_pixels, double *__restrict__ partials) {
    const int b = blockIdx.y;
    const pixel_t K = (pixel_t)shifts[b];
    const pixel_t *v = values + (size_t)b * n_pixels;
    pixel_t acc[N_SUMS];
    for (int k = 0; k < N_SUMS; k++) acc[k] = (pixel_t)0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pixels;
         i += gridDim.x * blockDim.x) {
        const pixel_t x = xi_x[i], y = xi_y[i];
        const pixel_t vp = v[i] - K;
        const pixel_t a = gradient_x[i] * vp;
        const pixel_t bb = gradient_y[i] * vp;
        const pixel_t p = a * x + bb * y;
        acc[0] += vp;
        acc[1] += vp * vp;
        acc[2] += a * x;
        acc[3] += a * y;
        acc[4] += a;
        acc[5] += bb * x;
        acc[6] += bb * y;
        acc[7] += bb;
        acc[8] += p * x;
        acc[9] += p * y;
    }
    __shared__ double tree[N_SUMS][TREE];
    for (int k = 0; k < N_SUMS; k++) tree[k][threadIdx.x] = (double)acc[k];
    __syncthreads();
    for (int s = TREE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            for (int k = 0; k < N_SUMS; k++)
                tree[k][threadIdx.x] += tree[k][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < N_SUMS)
        partials[((size_t)b * gridDim.x + blockIdx.x) * N_SUMS + threadIdx.x] =
            tree[threadIdx.x][0];
}

// One pass of the two-pass final criterion (D2.7), f64 partials:
// mode 0 sums vp, mode 1 sums (vp - mean)^2, mode 2 sums
// (w * (ref - (vp - mean) / norm))^2
extern "C" __global__ void hrebsd_criterion_pass(
    const pixel_t *__restrict__ values, const double *__restrict__ shifts,
    const double *__restrict__ mean, const double *__restrict__ norm,
    const pixel_t *__restrict__ reference, const pixel_t *__restrict__ weights,
    const int has_weights, const int mode, const int n_pixels,
    double *__restrict__ partials) {
    const int b = blockIdx.y;
    const pixel_t K = (pixel_t)shifts[b];
    const pixel_t *v = values + (size_t)b * n_pixels;
    const double mb = mode > 0 ? mean[b] : 0.0;
    const double nb = mode > 1 ? norm[b] : 1.0;
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pixels;
         i += gridDim.x * blockDim.x) {
        const double vp = (double)(v[i] - K);
        if (mode == 0) {
            acc += vp;
        } else {
            const double centred = vp - mb;
            if (mode == 1) {
                acc += centred * centred;
            } else {
                double r = (double)reference[i] - centred / nb;
                if (has_weights) r *= (double)weights[i];
                acc += r * r;
            }
        }
    }
    __shared__ double tree[TREE];
    tree[threadIdx.x] = acc;
    __syncthreads();
    for (int s = TREE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) tree[threadIdx.x] += tree[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[(size_t)b * gridDim.x + blockIdx.x] = tree[0];
}

// The fixed-order cross-block pass: grid (B), one thread per column
// sums the block partials of one pattern in block order
extern "C" __global__ void hrebsd_reduce_partials(
    const double *__restrict__ partials, const int n_blocks, const int width,
    double *__restrict__ out) {
    const int b = blockIdx.x;
    const int k = threadIdx.x;
    if (k >= width) return;
    const double *p = partials + (size_t)b * n_blocks * width;
    double acc = 0.0;
    for (int j = 0; j < n_blocks; j++) acc += p[(size_t)j * width + k];
    out[(size_t)b * width + k] = acc;
}

// One IC-GN step per ACTIVE pattern, f64 throughout (D21.4, D21.6):
// the algebra and order of ``_batched.solve_update``
extern "C" __global__ void hrebsd_solve_update(
    const double *__restrict__ sums, const double *__restrict__ constants,
    const double *__restrict__ upper, const double *__restrict__ corners,
    const double gradient_scale, const double n_pixels, const double min_step,
    const double step_scale, const long long max_iterations, const int n_slots,
    double *matrices, double *shifts, long long *iterations, double *norm_dp,
    bool *active, bool *converged, bool *failed) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_slots || !active[b]) return;
    const double *s = sums + (size_t)b * N_SUMS;
    const double mean = s[0] / n_pixels;
    const double norm = sqrt(s[1] - s[0] * mean);
    bool bad = !(isfinite(norm) && norm > 0.0);
    const double projected[8] = {s[2], s[3], s[4], s[5], s[6], s[7], -s[8], -s[9]};
    double rhs[8], z[8], dp[8];
    for (int k = 0; k < 8; k++) {
        const double centred = projected[k] - mean * constants[8 + k];
        rhs[k] = -(gradient_scale * (constants[k] - centred / norm));
    }
    for (int i = 0; i < 8; i++) {
        double acc = rhs[i];
        for (int j = 0; j < i; j++) acc = acc - upper[j * 8 + i] * z[j];
        z[i] = acc / upper[i * 8 + i];
    }
    for (int i = 7; i >= 0; i--) {
        double acc = z[i];
        for (int j = i + 1; j < 8; j++) acc = acc - upper[i * 8 + j] * dp[j];
        dp[i] = acc / upper[i * 8 + i];
    }
    for (int k = 0; k < 8; k++) {
        dp[k] = dp[k] * step_scale;
        if (!isfinite(dp[k])) bad = true;
    }
    // The D2.5 corner norm; the maximum PROPAGATES NaN (D21.6.1)
    double nd = 0.0;
    bool any_nan = false;
    for (int q = 0; q < 4; q++) {
        const double x = corners[2 * q], y = corners[2 * q + 1];
        const double sc = dp[6] * x + dp[7] * y + 1.0;
        const double xw = ((1 + dp[0]) * x + dp[1] * y + dp[2]) / sc;
        const double yw = (dp[3] * x + (1 + dp[4]) * y + dp[5]) / sc;
        const double d = hypot(xw - x, yw - y);
        if (isnan(d)) any_nan = true;
        else if (q == 0 || d > nd) nd = d;
    }
    if (any_nan) nd = HREBSD_NAN;
    // W(dp), its closed-form inverse, W <- W . W(dp)^-1, W33
    const double d00 = 1 + dp[0], d01 = dp[1], d02 = dp[2];
    const double d10 = dp[3], d11 = 1 + dp[4], d12 = dp[5];
    const double d20 = dp[6], d21 = dp[7], d22 = 1.0;
    const double c00 = d11 * d22 - d12 * d21;
    const double c01 = d12 * d20 - d10 * d22;
    const double c02 = d10 * d21 - d11 * d20;
    const double det = d00 * c00 + d01 * c01 + d02 * c02;
    if (!(isfinite(det) && det != 0.0)) bad = true;
    double inv[9] = {c00, d02 * d21 - d01 * d22, d01 * d12 - d02 * d11,
                     c01, d00 * d22 - d02 * d20, d02 * d10 - d00 * d12,
                     c02, d01 * d20 - d00 * d21, d00 * d11 - d01 * d10};
    for (int k = 0; k < 9; k++) inv[k] = inv[k] / det;
    double *m = matrices + 9 * (size_t)b;
    double product[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            product[3 * i + j] = m[3 * i] * inv[j] + m[3 * i + 1] * inv[3 + j] +
                                 m[3 * i + 2] * inv[6 + j];
    const double w33 = product[8];
    if (!(isfinite(w33) && w33 != 0.0)) bad = true;
    if (bad) {
        failed[b] = true;
        active[b] = false;
        norm_dp[b] = HREBSD_NAN;
        return;
    }
    for (int k = 0; k < 9; k++) m[k] = product[k] / w33;
    double next = shifts[b] + mean;
#if HREBSD_MIXED
    next = (double)(float)next;
#endif
    shifts[b] = next;
    const long long it = iterations[b] + 1;
    iterations[b] = it;
    norm_dp[b] = nd;
    if (nd < min_step) {
        converged[b] = true;
        active[b] = false;
    } else if (it >= max_iterations) {
        active[b] = false;
    }
}
"""

# Every kernel of :data:`_SOURCE`, constructed per call
_KERNEL_NAMES: tuple[str, ...] = (
    "hrebsd_gather",
    "hrebsd_pixel_sums",
    "hrebsd_criterion_pass",
    "hrebsd_reduce_partials",
    "hrebsd_solve_update",
)


# The order-3 B-spline prefilter of the device targets (D21.3), one
# thread per line in float64: the algebraic mirror-mode recursion (the
# pole, the gain, the causal and the anticausal initialisation) of
# ``scipy.ndimage.spline_filter1d``, NOT scipy's order of operations.
# It equals scipy's coefficients up to f64 rounding (RMS about 7e-14;
# about 1 in 1e7 of the f32-cast coefficients differs by one ulp),
# measured with and without fused multiply-add contraction alike
# (validation.md V9 ledger 108), so it compiles with the NVRTC default
# ``--fmad`` like every other kernel (D21.4)
_SPLINE_POLE = math.sqrt(3.0) - 2.0
_SPLINE_GAIN = (1.0 - _SPLINE_POLE) * (1.0 - 1.0 / _SPLINE_POLE)
_SPLINE_THREADS = 128
_SPLINE_SOURCE = r"""
extern "C" __global__ void hrebsd_spline_prefilter(
    double *data, const int n_lines, const int length,
    const long long pattern_step, const long long line_step,
    const long long element_step, const double gain, const double z,
    const double z_n_1) {
    const int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n_lines) return;
    double *c = data + (long long)blockIdx.y * pattern_step +
                (long long)line * line_step;
    const long long s = element_step;
    for (int i = 0; i < length; ++i) c[i * s] *= gain;
    // causal initialisation, mirror boundary
    double z_i = z;
    double c0 = c[0] + z_n_1 * c[(length - 1) * s];
    for (int i = 1; i < length - 1; ++i) {
        c0 += z_i * (c[i * s] + z_n_1 * c[(length - 1 - i) * s]);
        z_i *= z;
    }
    c[0] = c0 / (1.0 - z_n_1 * z_n_1);
    for (int i = 1; i < length; ++i) c[i * s] += z * c[(i - 1) * s];
    // anticausal initialisation, mirror boundary
    c[(length - 1) * s] =
        (z * c[(length - 2) * s] + c[(length - 1) * s]) * z / (z * z - 1.0);
    for (int i = length - 2; i >= 0; --i)
        c[i * s] = z * (c[(i + 1) * s] - c[i * s]);
}
"""


def make_cupy_kernel_namespace(device_precision: str):
    """Return the cupy :class:`~kikuchipy.indexing._hrebsd._batched.
    KernelNamespace` at *device_precision* (D21.9.5): every kernel of
    :data:`_KERNEL_NAMES` CONSTRUCTED here through ``cupy.RawKernel``
    on every call, no fast math (D21.4)."""
    import cupy

    mixed = device_precision == "mixed"
    pixel_dtype = cupy.float32 if mixed else cupy.float64
    # The shared-memory trees are as wide as a block of the layout
    tree = f"#define TREE {_batched.LAUNCH_THREADS}\n"
    code = _PREAMBLE[device_precision] + tree + _SOURCE
    kernels = {
        name: cupy.RawKernel(code, name, options=_KERNEL_OPTIONS)
        for name in _KERNEL_NAMES
    }
    spline_kernel = cupy.RawKernel(
        _SPLINE_SOURCE, "hrebsd_spline_prefilter", options=_KERNEL_OPTIONS
    )

    def layout(resident):
        threads, blocks, _ = _batched._launch_layout(resident.n_pixels)
        return threads, blocks

    def reduce_partials(partials, n_slots, n_blocks, width):
        out = cupy.empty((n_slots, width), dtype=cupy.float64)
        kernels["hrebsd_reduce_partials"](
            (n_slots,),
            (max(width, 1),),
            (partials, np.int32(n_blocks), np.int32(width), out),
        )
        return out

    def gather(resident, coefficients, matrices):
        coefficients = cupy.ascontiguousarray(
            cupy.asarray(coefficients, dtype=cupy.float32)
        )
        matrices = cupy.ascontiguousarray(cupy.asarray(matrices, dtype=cupy.float64))
        n_slots = int(matrices.shape[0])
        n_rows, n_cols = resident.shape
        threads, blocks = layout(resident)
        values = cupy.empty((n_slots, resident.n_pixels), dtype=pixel_dtype)
        bad = cupy.empty((n_slots, blocks), dtype=cupy.int32)
        kernels["hrebsd_gather"](
            (blocks, n_slots),
            (threads,),
            (
                coefficients,
                np.int32(n_rows),
                np.int32(n_cols),
                resident.xi_x,
                resident.xi_y,
                # the f64 build never reads the pixel indices
                resident.mask_index if resident.columns is None else resident.columns,
                resident.mask_index if resident.rows is None else resident.rows,
                np.int32(resident.n_pixels),
                matrices,
                np.float64(resident.offset[0]),
                np.float64(resident.offset[1]),
                values,
                bad,
            ),
        )
        return values, ~cupy.any(bad != 0, axis=1)

    def pixel_sums(resident, values, shifts):
        values = cupy.ascontiguousarray(cupy.asarray(values, dtype=pixel_dtype))
        shifts = cupy.ascontiguousarray(cupy.asarray(shifts, dtype=cupy.float64))
        n_slots = int(values.shape[0])
        threads, blocks = layout(resident)
        partials = cupy.empty((n_slots, blocks, _batched.N_SUMS), dtype=cupy.float64)
        kernels["hrebsd_pixel_sums"](
            (blocks, n_slots),
            (threads,),
            (
                values,
                shifts,
                resident.xi_x,
                resident.xi_y,
                resident.gradient_x,
                resident.gradient_y,
                np.int32(resident.n_pixels),
                partials,
            ),
        )
        return reduce_partials(partials, n_slots, blocks, _batched.N_SUMS)

    def final_criterion(resident, values, shifts):
        values = cupy.ascontiguousarray(cupy.asarray(values, dtype=pixel_dtype))
        shifts = cupy.ascontiguousarray(cupy.asarray(shifts, dtype=cupy.float64))
        n_slots = int(values.shape[0])
        threads, blocks = layout(resident)
        weights = resident.weights
        has_weights = weights is not None
        if not has_weights:
            weights = resident.reference
        mean = cupy.zeros(n_slots, dtype=cupy.float64)
        norm = cupy.ones(n_slots, dtype=cupy.float64)
        partials = cupy.empty((n_slots, blocks, 1), dtype=cupy.float64)
        results = []
        for mode in range(3):
            kernels["hrebsd_criterion_pass"](
                (blocks, n_slots),
                (threads,),
                (
                    values,
                    shifts,
                    mean,
                    norm,
                    resident.reference,
                    weights,
                    np.int32(has_weights),
                    np.int32(mode),
                    np.int32(resident.n_pixels),
                    partials,
                ),
            )
            total = reduce_partials(partials, n_slots, blocks, 1)[:, 0]
            if mode == 0:
                mean = cupy.ascontiguousarray(total / float(resident.n_pixels))
            elif mode == 1:
                norm = cupy.ascontiguousarray(cupy.sqrt(total))
            results.append(total)
        criterion = results[2]
        bad = ~(cupy.isfinite(norm) & (norm > 0))
        return cupy.where(bad, cupy.nan, criterion)

    def reduce_solve_update(resident, sums, lockstep, options):
        sums = cupy.ascontiguousarray(cupy.asarray(sums, dtype=cupy.float64))
        n_slots = int(sums.shape[0])
        for name, dtype in (
            ("matrices", cupy.float64),
            ("shifts", cupy.float64),
            ("iterations", cupy.int64),
            ("norm_dp", cupy.float64),
            ("active", cupy.bool_),
            ("converged", cupy.bool_),
            ("failed", cupy.bool_),
        ):
            array = getattr(lockstep, name)
            if not (
                isinstance(array, cupy.ndarray)
                and array.dtype == dtype
                and array.flags.c_contiguous
            ):
                setattr(
                    lockstep, name, cupy.ascontiguousarray(cupy.asarray(array, dtype))
                )
        # One single-thread block per pattern: the per-pattern launch
        # geometry is independent of B (D21.7.3)
        kernels["hrebsd_solve_update"](
            (n_slots,),
            (1,),
            (
                sums,
                resident.constants,
                resident.cholesky,
                resident.corners,
                np.float64(resident.gradient_scale),
                np.float64(resident.n_pixels),
                np.float64(options["min_step"]),
                np.float64(options["step_scale"]),
                np.int64(options["max_iterations"]),
                np.int32(n_slots),
                lockstep.matrices,
                lockstep.shifts,
                lockstep.iterations,
                lockstep.norm_dp,
                lockstep.active,
                lockstep.converged,
                lockstep.failed,
            ),
        )

    def spline_prefilter(data, axis):
        """Apply the order-3 mirror-mode B-spline prefilter IN PLACE
        along *axis* (1 or 2) of the C-contiguous ``(P, nrows, ncols)``
        float64 device array *data* (D21.3).  The grid is ``(lines per
        pattern / threads, P)``: per pattern a function of the pattern
        geometry only, never of B (D21.7.3)."""
        n_patterns, n_rows, n_cols = (int(i) for i in data.shape)
        if axis == 1:
            length, n_lines, line_step, element_step = n_rows, n_cols, 1, n_cols
        else:
            length, n_lines, line_step, element_step = n_cols, n_rows, n_cols, 1
        if length < 2:
            return
        blocks = (n_lines + _SPLINE_THREADS - 1) // _SPLINE_THREADS
        spline_kernel(
            (blocks, n_patterns),
            (_SPLINE_THREADS,),
            (
                data,
                np.int32(n_lines),
                np.int32(length),
                np.int64(n_rows * n_cols),
                np.int64(line_step),
                np.int64(element_step),
                np.float64(_SPLINE_GAIN),
                np.float64(_SPLINE_POLE),
                np.float64(math.pow(_SPLINE_POLE, length - 1)),
            ),
        )

    namespace = _batched.KernelNamespace(
        cupy,
        cupy.fft,
        device_precision,
        cupy.cuda.memory.OutOfMemoryError,
        gather,
        pixel_sums,
        reduce_solve_update,
        final_criterion,
    )
    # Beside the four D21.9.5 entry points: the device spline
    # prefilter of the target preprocessing (D21.3), its kernel built
    # here with the others
    namespace.spline_prefilter = spline_prefilter
    return namespace
