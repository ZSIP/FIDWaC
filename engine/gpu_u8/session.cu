// Device-resident GPU session for the uint8 batched search (cm=5 / cm=6).
//
// The session uploads the zigzag coefficients and centred blocks of up to 3 channels (and the RGB
// originals for cm=6) ONCE, allocates its work buffers ONCE, and keeps the quantised coefficients
// of the last search of every channel in VRAM, so the cm=6 RGB check reads them without a copy.
// Per call only the row indices and per-row parameters go to the device, and only the results
// come back.
//
// Fused search kernel per row: binary search (float criterion) + backscan (integer criterion)
// -> full integer verification with a basis-sum IDCT -> quantisation. The basis-sum IDCT matches
// scipy's except at exact rounding ties; a row that meets a tie (|frac - 0.5| < 1e-9), or whose L2
// precheck sits within 1e-5 of the threshold, is returned with FLAG = 1 and recomputed on the CPU
// with scipy by the caller. Arithmetic mirrors engine/u8_lockstep.py; build with -fmad=false.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef BSZ
#define BSZ 8  // block edge; build.sh compiles one library per edge (8, 16, 32)
#endif
#define NN (BSZ * BSZ)
#define TOTAL NN
#define NWARP ((NN + 31) / 32)
#define CU8 128.0
#define LO8 0.0
#define HI8 255.0
__constant__ int g_break_after = 3;  // backscan_break_after, set by the host before each launch
#define MAX_VERIFY 5
#define MAX_QUANT 16
#define TIE_TOL 1e-9
#define PRECHECK_TOL 1e-5

// ---------------------------------------------------------------------------------------------
// device helpers (same arithmetic as engine/u8_lockstep.py)
// ---------------------------------------------------------------------------------------------
__device__ double tail_l2(const double* zz, int L) {
    double s = 0.0;
    for (int k = L; k < NN; ++k) {
        float t = (float)zz[k];
        s += (double)(t * t);
    }
    return sqrt(s);
}

__device__ double eval_float(const double* zz, const double* orig, int L, double acc,
                             const double* basis2) {
    if (tail_l2(zz, L) > acc * (double)BSZ) return acc + 1.0;
    double rec[NN];
    for (int p = 0; p < NN; ++p) rec[p] = 0.0;
    for (int k = 0; k < L; ++k) {
        double ck = zz[k];
        for (int p = 0; p < NN; ++p) rec[p] += ck * basis2[k * NN + p];
    }
    double mx = 0.0;
    for (int p = 0; p < NN; ++p) {
        double d = orig[p] - rec[p];
        if (d < 0.0) d = -d;
        if (d > mx) mx = d;
    }
    return mx;
}

__device__ double eval_quant(const double* dqz, const double* orig, int L, const double* basis2) {
    double rec[NN];
    for (int p = 0; p < NN; ++p) rec[p] = 0.0;
    for (int k = 0; k < L; ++k) {
        double ck = dqz[k];
        for (int p = 0; p < NN; ++p) rec[p] += ck * basis2[k * NN + p];
    }
    double mx = 0.0;
    for (int p = 0; p < NN; ++p) {
        double v = rint(rec[p] + CU8);
        if (v < LO8) v = LO8;
        else if (v > HI8) v = HI8;
        double d = orig[p] + CU8 - v;
        if (d < 0.0) d = -d;
        if (d > mx) mx = d;
    }
    return mx;
}

__device__ __forceinline__ bool is_tie(double x) {
    double f = x - floor(x);
    return fabs(f - 0.5) < TIE_TOL;
}

// ---------------------------------------------------------------------------------------------
// cm=6 pre-verification RGB error (same as rgb_pre.cu), reading the quantised coefficients of the
// last search of each channel straight from VRAM
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ double clip_u8(double v) {
    if (v < 0.0) return 0.0;
    if (v > 255.0) return 255.0;
    return v;
}

__device__ __forceinline__ double rnd(double x) {
    double r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2.0) != 0.0) r -= 1.0;
    return r;
}

__device__ void plane_from_res(const int64_t* res, double sf, const double* basis2, double* plane) {
    double cdq[NN];
    for (int k = 0; k < NN; ++k) cdq[k] = (double)res[k] / sf;
    for (int p = 0; p < NN; ++p) {
        double rec = 0.0;
        for (int k = 0; k < NN; ++k) rec += cdq[k] * basis2[k * NN + p];
        plane[p] = clip_u8(rec + 128.0);
    }
}

__global__ void rgb_kernel(const int64_t* __restrict__ RY, const int64_t* __restrict__ RCB,
                           const int64_t* __restrict__ RCR, const float* __restrict__ RF,
                           const float* __restrict__ GF, const float* __restrict__ BF,
                           const int64_t* __restrict__ idx, int n,
                           const double* __restrict__ basis2, double sf,
                           double* __restrict__ err, int32_t* __restrict__ flag) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int64_t r = idx[t];
    double Y[NN], Cb[NN], Cr[NN];
    plane_from_res(RY + (size_t)t * NN, sf, basis2, Y);
    plane_from_res(RCB + (size_t)t * NN, sf, basis2, Cb);
    plane_from_res(RCR + (size_t)t * NN, sf, basis2, Cr);
    double mx = 0.0;
    int tie = 0;
    for (int p = 0; p < NN; ++p) {
        double cbm = Cb[p] - 128.0;
        double crm = Cr[p] - 128.0;
        double xr = Y[p] + 1.403 * crm;
        double xg = Y[p] - 0.344 * cbm - 0.714 * crm;
        double xb = Y[p] + 1.773 * cbm;
        tie |= is_tie(xr) | is_tie(xg) | is_tie(xb);
        double R = clip_u8(rnd(xr));
        double G = clip_u8(rnd(xg));
        double Bc = clip_u8(rnd(xb));
        double dr = fabs((double)RF[(size_t)r * NN + p] - R);
        double dg = fabs((double)GF[(size_t)r * NN + p] - G);
        double db = fabs((double)BF[(size_t)r * NN + p] - Bc);
        if (dr > mx) mx = dr;
        if (dg > mx) mx = dg;
        if (db > mx) mx = db;
    }
    err[t] = mx;
    flag[t] = tie;
}

__global__ void scatter_rows(int64_t* __restrict__ dst, const int64_t* __restrict__ pos,
                             const int64_t* __restrict__ rows, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m * NN) return;
    int a = j / NN, k = j % NN;
    dst[(size_t)pos[a] * NN + k] = rows[(size_t)a * NN + k];
}

// ---------------------------------------------------------------------------------------------
// session
// ---------------------------------------------------------------------------------------------
struct Session {
    int B, nch;
    double* zz[3];
    double* org[3];
    double* basis;
    float *rf, *gf, *bf;
    int64_t* idx;
    double *acc, *sf;
    int64_t* bs;
    int64_t* lf;
    double* err;
    int32_t* flag;
    int64_t* res[3];
    int64_t *pos, *rows;
};

static void sess_release(Session* s) {
    if (!s) return;
    for (int c = 0; c < 3; ++c) {
        cudaFree(s->zz[c]);
        cudaFree(s->org[c]);
        cudaFree(s->res[c]);
    }
    cudaFree(s->basis);
    cudaFree(s->rf);
    cudaFree(s->gf);
    cudaFree(s->bf);
    cudaFree(s->idx);
    cudaFree(s->acc);
    cudaFree(s->sf);
    cudaFree(s->bs);
    cudaFree(s->lf);
    cudaFree(s->err);
    cudaFree(s->flag);
    cudaFree(s->pos);
    cudaFree(s->rows);
    free(s);
}

#define CK(call)                                          \
    do {                                                  \
        cudaError_t _e = (call);                          \
        if (_e != cudaSuccess) {                          \
            *rc = (int)_e;                                \
            sess_release(s);                              \
            return NULL;                                  \
        }                                                 \
    } while (0)

// zz/org: nch pointers to (B,64) float64 host arrays; rf/gf/bf: (B,64) float32 or NULL.
extern "C" void* sess_create(int B, int nch, const double* zz0, const double* org0,
                             const double* zz1, const double* org1, const double* zz2,
                             const double* org2, const double* basis2, const float* rf,
                             const float* gf, const float* bf, int* rc) {
    *rc = 0;
    Session* s = (Session*)calloc(1, sizeof(Session));
    if (!s) {
        *rc = -1;
        return NULL;
    }
    s->B = B;
    s->nch = nch;
    const double* zz[3] = {zz0, zz1, zz2};
    const double* org[3] = {org0, org1, org2};
    size_t nn = (size_t)B * NN;
    for (int c = 0; c < nch; ++c) {
        CK(cudaMalloc(&s->zz[c], sizeof(double) * nn));
        CK(cudaMalloc(&s->org[c], sizeof(double) * nn));
        CK(cudaMalloc(&s->res[c], sizeof(int64_t) * nn));
        CK(cudaMemcpy(s->zz[c], zz[c], sizeof(double) * nn, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s->org[c], org[c], sizeof(double) * nn, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&s->basis, sizeof(double) * NN * NN));
    CK(cudaMemcpy(s->basis, basis2, sizeof(double) * NN * NN, cudaMemcpyHostToDevice));
    if (rf && gf && bf) {
        CK(cudaMalloc(&s->rf, sizeof(float) * nn));
        CK(cudaMalloc(&s->gf, sizeof(float) * nn));
        CK(cudaMalloc(&s->bf, sizeof(float) * nn));
        CK(cudaMemcpy(s->rf, rf, sizeof(float) * nn, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s->gf, gf, sizeof(float) * nn, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s->bf, bf, sizeof(float) * nn, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&s->idx, sizeof(int64_t) * B));
    CK(cudaMalloc(&s->acc, sizeof(double) * B));
    CK(cudaMalloc(&s->sf, sizeof(double) * B));
    CK(cudaMalloc(&s->bs, sizeof(int64_t) * B));
    CK(cudaMalloc(&s->lf, sizeof(int64_t) * B));
    CK(cudaMalloc(&s->err, sizeof(double) * B));
    CK(cudaMalloc(&s->flag, sizeof(int32_t) * B));
    CK(cudaMalloc(&s->pos, sizeof(int64_t) * B));
    CK(cudaMalloc(&s->rows, sizeof(int64_t) * nn));
    return s;
}

extern "C" void sess_free(void* h) { sess_release((Session*)h); }

#define CKR(call)                                         \
    do {                                                  \
        cudaError_t _e = (call);                          \
        if (_e != cudaSuccess) return (int)_e;            \
    } while (0)

// Overwrite rows (positions pos[0..m) of the last search of channel ch) with CPU results.
extern "C" int sess_patch(void* h, int ch, const int64_t* pos, int m, const int64_t* rows) {
    Session* s = (Session*)h;
    if (!s || ch < 0 || ch >= s->nch || m < 0 || m > s->B) return -2;
    if (m == 0) return 0;
    CKR(cudaMemcpy(s->pos, pos, sizeof(int64_t) * m, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(s->rows, rows, sizeof(int64_t) * (size_t)m * NN, cudaMemcpyHostToDevice));
    int threads = 256;
    int grid = (m * NN + threads - 1) / threads;
    scatter_rows<<<grid, threads>>>(s->res[ch], s->pos, s->rows, m);
    CKR(cudaGetLastError());
    CKR(cudaDeviceSynchronize());
    return 0;
}

// cm=6 pre-verify RGB error for rows idx[0..n), using the last search of channels 0, 1, 2 (which
// must have been run on the same idx, in the same order).
extern "C" int sess_rgb(void* h, const int64_t* idx, int n, double sf, double* err,
                        int32_t* flag) {
    Session* s = (Session*)h;
    if (!s || s->nch < 3 || !s->rf || n < 0 || n > s->B) return -2;
    if (n == 0) return 0;
    CKR(cudaMemcpy(s->idx, idx, sizeof(int64_t) * n, cudaMemcpyHostToDevice));
    int threads = 128;
    int grid = (n + threads - 1) / threads;
    rgb_kernel<<<grid, threads>>>(s->res[0], s->res[1], s->res[2], s->rf, s->gf, s->bf, s->idx, n,
                                  s->basis, sf, s->err, s->flag);
    CKR(cudaGetLastError());
    CKR(cudaMemcpy(err, s->err, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CKR(cudaMemcpy(flag, s->flag, sizeof(int32_t) * n, cudaMemcpyDeviceToHost));
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Pixel-parallel search: one block of NN threads per row, thread p owns pixel p.
// Per pixel the sums run over k in the same ascending order as in search_kernel, the scalar
// decisions (tail energy, integer backscan steps, verification steps) are taken by thread 0 and
// shared through shared memory, and maxima/minima are order independent, so the results are
// bit-identical to search_kernel.
// ---------------------------------------------------------------------------------------------
// Block maximum / minimum of one value per thread (NN threads = NN / 32 warps): a warp shuffle
// reduction, then the two warp results meet in shared memory. Max and min do not depend on the
// order, so the result is exact.
__device__ double px_red_max(double v, double* s) {
    int t = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1) {
        double o = __shfl_down_sync(0xffffffffu, v, off);
        v = v > o ? v : o;
    }
    if ((t & 31) == 0) s[t >> 5] = v;
    __syncthreads();
    double r = s[0];
    for (int w = 1; w < NWARP; ++w) r = r > s[w] ? r : s[w];
    __syncthreads();
    return r;
}

__device__ double px_red_min(double v, double* s) {
    int t = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1) {
        double o = __shfl_down_sync(0xffffffffu, v, off);
        v = v < o ? v : o;
    }
    if ((t & 31) == 0) s[t >> 5] = v;
    __syncthreads();
    double r = s[0];
    for (int w = 1; w < NWARP; ++w) r = r < s[w] ? r : s[w];
    __syncthreads();
    return r;
}

// sum_{k<L} coef[k] * basis[k, t], k ascending (the pixel t of the reconstruction)
__device__ double px_rec(const double* coef, int L, const double* basis2) {
    int t = threadIdx.x;
    double rec = 0.0;
    for (int k = 0; k < L; ++k) rec += coef[k] * basis2[k * NN + t];
    return rec;
}

// same, with the float32 cast of the coefficient used by the backscan start
__device__ double px_rec_f(const double* coef, int L, const double* basis2) {
    int t = threadIdx.x;
    double rec = 0.0;
    for (int k = 0; k < L; ++k) rec += (double)(float)coef[k] * basis2[k * NN + t];
    return rec;
}

// Classic verification of length L (_eval_error_for_len, classic path). Returns the
// error; *flag is set when the row needs the CPU (rounding tie or precheck boundary) and
// *fail when the L2 precheck rejects the length. All threads take the same branch.
__device__ double px_classic(const double* sz, const double* so, int L, double acc,
                             const double* basis2, double* sred, int* sflag, int* sfail) {
    int t = threadIdx.x;
    if (t == 0) {
        double tl = tail_l2(sz, L);
        double thr = acc * (double)BSZ;
        double win = PRECHECK_TOL * (thr > 1.0 ? thr : 1.0);
        *sfail = 0;
        if (fabs(tl - thr) <= win) *sflag = 1;
        else if (tl > thr) *sfail = 1;
    }
    __syncthreads();
    if (*sflag) return 0.0;
    if (*sfail) return acc + 1.0;
    double rec = px_rec(sz, L, basis2);
    double tt = rec + CU8;
    double tie = is_tie(tt) ? 1.0 : 0.0;
    double any_tie = px_red_max(tie, sred);
    if (any_tie > 0.0) {
        if (t == 0) *sflag = 1;
        __syncthreads();
        return 0.0;
    }
    double v = rint(tt);
    if (v < LO8) v = LO8;
    else if (v > HI8) v = HI8;
    double e = (so[t] + CU8) - v;
    double mx = px_red_max(e > 0.0 ? e : 0.0, sred);
    double mn = px_red_min(e < 0.0 ? e : 0.0, sred);
    return mx > -mn ? mx : -mn;
}

__global__ void __launch_bounds__(NN) search_kernel(const double* __restrict__ ZZ, const double* __restrict__ ORIG,
                                 const double* __restrict__ basis2, const int64_t* __restrict__ idx,
                                 int n, const double* __restrict__ ACC,
                                 const double* __restrict__ SF, const int64_t* __restrict__ BS,
                                 int64_t* __restrict__ LF, double* __restrict__ ERR,
                                 int64_t* __restrict__ RES, int32_t* __restrict__ FLAG) {
    __shared__ double sz[NN], so[NN], sq[NN], sred[NN], sdiff[NN];
    __shared__ double sacc, ssf;
    __shared__ int64_t sbs;
    __shared__ int slo, shi, sans, smid, sstart, snewL, sbrk, sfail, sflag, sfl;
    int t = threadIdx.x;
    int pos = blockIdx.x;
    if (pos >= n) return;  // uniform for the whole block
    int64_t r = idx[pos];
    sz[t] = ZZ[(size_t)r * NN + t];
    so[t] = ORIG[(size_t)r * NN + t];
    if (t == 0) {
        sacc = ACC[pos];
        ssf = SF[pos];
        sbs = BS[pos];
        sflag = 0;
        sfail = 0;
        slo = 1;
        shi = TOTAL;
        sans = TOTAL;
    }
    __syncthreads();
    double acc = sacc;
    double sf = ssf;

    // 1. binary search on [1, NN] with the float criterion (eval_float)
    while (true) {
        if (t == 0) smid = (slo <= shi) ? (slo + shi) / 2 : -1;
        __syncthreads();
        if (smid < 0) break;
        double err;
        if (t == 0) sfl = tail_l2(sz, smid) > acc * (double)BSZ;
        __syncthreads();
        if (sfl) {
            err = acc + 1.0;
        } else {
            double rec = px_rec(sz, smid, basis2);
            double d = so[t] - rec;
            d = d < 0.0 ? -d : d;
            err = px_red_max(d, sred);
        }
        if (t == 0) {
            if (err <= acc) {
                sans = smid;
                shi = smid - 1;
            } else {
                slo = smid + 1;
            }
        }
        __syncthreads();
    }

    // 2. integer backscan
    if (t == 0) snewL = sans;
    __syncthreads();
    if (sbs > 0 && sans > 1) {
        if (t == 0) {
            int start = sans - (int)sbs;
            sstart = start < 1 ? 1 : start;
            sbrk = 0;
            sfail = 0;
        }
        __syncthreads();
        double rec = px_rec_f(sz, sans, basis2);
        sdiff[t] = so[t] - rec;
        __syncthreads();
        for (int k = sans - 1; k >= sstart; --k) {
            double ck = (double)(float)sz[k];
            sdiff[t] += basis2[k * NN + t] * ck;
            double tv = sdiff[t] + so[t] + CU8;
            double v = rint(tv);
            if (v < LO8) v = LO8;
            else if (v > HI8) v = HI8;
            double e = (so[t] + CU8) - v;
            double mx = px_red_max(e > 0.0 ? e : 0.0, sred);
            double mn = px_red_min(e < 0.0 ? e : 0.0, sred);
            if (t == 0) {
                double e2 = mx > -mn ? mx : -mn;
                if (e2 <= acc) {
                    snewL = k;
                    sfail = 0;
                } else {
                    sfail += 1;
                    if (g_break_after > 0 && sfail >= g_break_after) sbrk = 1;
                }
            }
            __syncthreads();
            if (sbrk) break;
        }
    }
    int L = snewL;

    // 3. classic verification (up to MAX_VERIFY increments)
    double e_full = px_classic(sz, so, L, acc, basis2, sred, &sflag, &sfail);
    int steps = 0;
    while (!sflag && e_full > acc && L < TOTAL && steps < MAX_VERIFY) {
        L += 1;
        steps += 1;
        e_full = px_classic(sz, so, L, acc, basis2, sred, &sflag, &sfail);
    }

    if (sflag) {  // the CPU recomputes this row
        if (t == 0) {
            LF[pos] = (int64_t)L;
            ERR[pos] = 0.0;
            FLAG[pos] = 1;
        }
        RES[(size_t)pos * NN + t] = 0;
        return;
    }

    // 4. quantisation and the post-quantisation check (eval_quant)
    sq[t] = (t < L) ? rint(sz[t] * sf) / sf : 0.0;
    __syncthreads();
    double max_err;
    {
        double rec = px_rec(sq, L, basis2);
        double v = rint(rec + CU8);
        if (v < LO8) v = LO8;
        else if (v > HI8) v = HI8;
        double d = (so[t] + CU8) - v;
        d = d < 0.0 ? -d : d;
        max_err = px_red_max(d, sred);
    }
    int q = 0;
    while (max_err > acc && L < TOTAL && q < MAX_QUANT) {
        L += 1;
        q += 1;
        sq[t] = (t < L) ? rint(sz[t] * sf) / sf : 0.0;
        __syncthreads();
        double rec = px_rec(sq, L, basis2);
        double v = rint(rec + CU8);
        if (v < LO8) v = LO8;
        else if (v > HI8) v = HI8;
        double d = (so[t] + CU8) - v;
        d = d < 0.0 ? -d : d;
        max_err = px_red_max(d, sred);
    }

    if (t == 0) {
        LF[pos] = (int64_t)L;
        ERR[pos] = max_err;
        FLAG[pos] = 0;
    }
    RES[(size_t)pos * NN + t] = (t < L) ? (int64_t)rint(sz[t] * sf) : 0;
}

extern "C" int sess_search(void* h, int ch, const int64_t* idx, int n, const double* acc,
                              const double* sf, const int64_t* bs, int64_t* LF, double* ERR,
                              int64_t* RES, int32_t* FLAG, int brk) {
    Session* s = (Session*)h;
    if (!s || ch < 0 || ch >= s->nch || n < 0 || n > s->B) return -2;
    if (n == 0) return 0;
    CKR(cudaMemcpy(s->idx, idx, sizeof(int64_t) * n, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(s->acc, acc, sizeof(double) * n, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(s->sf, sf, sizeof(double) * n, cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(s->bs, bs, sizeof(int64_t) * n, cudaMemcpyHostToDevice));
    CKR(cudaMemcpyToSymbol(g_break_after, &brk, sizeof(int)));
    search_kernel<<<n, NN>>>(s->zz[ch], s->org[ch], s->basis, s->idx, n, s->acc, s->sf, s->bs,
                                s->lf, s->err, s->res[ch], s->flag);
    CKR(cudaGetLastError());
    CKR(cudaDeviceSynchronize());
    CKR(cudaMemcpy(LF, s->lf, sizeof(int64_t) * n, cudaMemcpyDeviceToHost));
    CKR(cudaMemcpy(ERR, s->err, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CKR(cudaMemcpy(RES, s->res[ch], sizeof(int64_t) * (size_t)n * NN, cudaMemcpyDeviceToHost));
    CKR(cudaMemcpy(FLAG, s->flag, sizeof(int32_t) * n, cudaMemcpyDeviceToHost));
    return 0;
}
