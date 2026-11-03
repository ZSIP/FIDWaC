// Batched uint8 accuracy search, phase kernels (one thread = one 8x8 block).
//
// Phase 1 (pre_kernel):   binary search of L on the float criterion + integer backscan.
// Phase 2 (host, scipy):  full integer verification with the reference IDCT, see
//                         engine/u8_lockstep.py::classic_verify (scipy rounds exact ties the way
//                         the decoder does; a basis sum cannot).
// Phase 3 (quant_kernel): quantisation and the post-quantisation check, starting from the L of
//                         phase 2.
//
// Arithmetic mirrors engine/u8_lockstep.py operation by operation (float64, same summation
// order, round-half-even). Build with -fmad=false so no FMA changes the rounding of the sums.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

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
#define MAX_QUANT 16

__device__ double tail_l2(const double* zz, int L) {
    double s = 0.0;
    for (int k = L; k < NN; ++k) {
        float t = (float)zz[k];
        s += (double)(t * t);
    }
    return sqrt(s);
}

// _nb_eval_fast, float mode, with the L2 precheck of _eval_float
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

// _nb_eval_fast, integer mode (decoder rounding and clipping) == _eval_quant
__device__ double eval_quant(const double* dqz, const double* orig, int L,
                             const double* basis2) {
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

// _int_err_from_diff: max(max e, -min e), e = (orig + 128) - clip(round(diff + orig + 128))
__device__ double int_err_from_diff(const double* diff, const double* orig) {
    double mx = 0.0, mn = 0.0;
    for (int p = 0; p < NN; ++p) {
        double t = diff[p] + orig[p] + CU8;
        double v = rint(t);
        if (v < LO8) v = LO8;
        else if (v > HI8) v = HI8;
        double e = (orig[p] + CU8) - v;
        if (e > mx) mx = e;
        if (e < mn) mn = e;
    }
    return mx > -mn ? mx : -mn;
}

__global__ void pre_kernel(const double* __restrict__ ZZ, const double* __restrict__ ORIG,
                           const double* __restrict__ ACC, const int64_t* __restrict__ BS,
                           const double* __restrict__ basis2, int64_t* __restrict__ LPRE, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const double* zz = ZZ + (size_t)i * NN;
    const double* orig = ORIG + (size_t)i * NN;
    double acc = ACC[i];
    int64_t bs = BS[i];

    int lo = 1, hi = TOTAL, ans = TOTAL;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        double err = eval_float(zz, orig, mid, acc, basis2);
        if (err <= acc) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    int L = ans;

    if (bs > 0 && L > 1) {
        int start = L - (int)bs;
        if (start < 1) start = 1;
        double diff[NN];
        for (int p = 0; p < NN; ++p) {
            double rec = 0.0;
            for (int k = 0; k < L; ++k) rec += (double)(float)zz[k] * basis2[k * NN + p];
            diff[p] = orig[p] - rec;
        }
        int new_L = L, fail = 0;
        for (int k = L - 1; k >= start; --k) {
            double ck = (double)(float)zz[k];
            for (int p = 0; p < NN; ++p) diff[p] += basis2[k * NN + p] * ck;
            double e2 = int_err_from_diff(diff, orig);
            if (e2 <= acc) {
                new_L = k;
                fail = 0;
            } else {
                fail += 1;
                if (g_break_after > 0 && fail >= g_break_after) break;
            }
        }
        L = new_L;
    }
    LPRE[i] = (int64_t)L;
}

__global__ void quant_kernel(const double* __restrict__ ZZ, const double* __restrict__ ORIG,
                             const double* __restrict__ ACC, const double* __restrict__ SF,
                             const double* __restrict__ basis2, const int64_t* __restrict__ LIN,
                             int64_t* __restrict__ LF, double* __restrict__ ERR,
                             int64_t* __restrict__ RES, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const double* zz = ZZ + (size_t)i * NN;
    const double* orig = ORIG + (size_t)i * NN;
    double acc = ACC[i];
    double sf = SF[i];
    int L = (int)LIN[i];

    double dqz[NN];
    for (int p = 0; p < NN; ++p) dqz[p] = 0.0;
    for (int k = 0; k < L; ++k) dqz[k] = rint(zz[k] * sf) / sf;
    double max_err = eval_quant(dqz, orig, L, basis2);
    int q = 0;
    while (max_err > acc && L < TOTAL && q < MAX_QUANT) {
        L += 1;
        q += 1;
        for (int p = 0; p < NN; ++p) dqz[p] = 0.0;
        for (int k = 0; k < L; ++k) dqz[k] = rint(zz[k] * sf) / sf;
        max_err = eval_quant(dqz, orig, L, basis2);
    }

    LF[i] = (int64_t)L;
    ERR[i] = max_err;
    for (int k = 0; k < L; ++k) RES[(size_t)i * NN + k] = (int64_t)rint(zz[k] * sf);
}

#define CHECK_CUDA(call)                                  \
    do {                                                  \
        cudaError_t _e = (call);                          \
        if (_e != cudaSuccess) {                          \
            rc = (int)_e;                                 \
            goto cleanup;                                 \
        }                                                 \
    } while (0)

// Phase 1 host entry. Returns 0 on success, otherwise the cudaError_t code.
extern "C" int pre_u8_gpu(const double* ZZ, const double* ORIG, const double* ACC,
                          const int64_t* BS, const double* basis2, int64_t* LPRE, int B, int brk) {
    int rc = 0;
    double *dZZ = NULL, *dORIG = NULL, *dACC = NULL, *dBA = NULL;
    int64_t *dBS = NULL, *dLPRE = NULL;
    size_t nzz = (size_t)B * NN;

    CHECK_CUDA(cudaMalloc(&dZZ, sizeof(double) * nzz));
    CHECK_CUDA(cudaMalloc(&dORIG, sizeof(double) * nzz));
    CHECK_CUDA(cudaMalloc(&dACC, sizeof(double) * B));
    CHECK_CUDA(cudaMalloc(&dBS, sizeof(int64_t) * B));
    CHECK_CUDA(cudaMalloc(&dBA, sizeof(double) * NN * NN));
    CHECK_CUDA(cudaMalloc(&dLPRE, sizeof(int64_t) * B));

    CHECK_CUDA(cudaMemcpy(dZZ, ZZ, sizeof(double) * nzz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dORIG, ORIG, sizeof(double) * nzz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dACC, ACC, sizeof(double) * B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBS, BS, sizeof(int64_t) * B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBA, basis2, sizeof(double) * NN * NN, cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int grid = (B + threads - 1) / threads;
        CHECK_CUDA(cudaMemcpyToSymbol(g_break_after, &brk, sizeof(int)));
        pre_kernel<<<grid, threads>>>(dZZ, dORIG, dACC, dBS, dBA, dLPRE, B);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemcpy(LPRE, dLPRE, sizeof(int64_t) * B, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(dZZ);
    cudaFree(dORIG);
    cudaFree(dACC);
    cudaFree(dBS);
    cudaFree(dBA);
    cudaFree(dLPRE);
    return rc;
}

// Phase 3 host entry. Returns 0 on success, otherwise the cudaError_t code.
extern "C" int quant_u8_gpu(const double* ZZ, const double* ORIG, const double* ACC,
                            const double* SF, const double* basis2, const int64_t* LIN,
                            int64_t* LF, double* ERR, int64_t* RES, int B) {
    int rc = 0;
    double *dZZ = NULL, *dORIG = NULL, *dACC = NULL, *dSF = NULL, *dBA = NULL, *dERR = NULL;
    int64_t *dLIN = NULL, *dLF = NULL, *dRES = NULL;
    size_t nzz = (size_t)B * NN;

    CHECK_CUDA(cudaMalloc(&dZZ, sizeof(double) * nzz));
    CHECK_CUDA(cudaMalloc(&dORIG, sizeof(double) * nzz));
    CHECK_CUDA(cudaMalloc(&dACC, sizeof(double) * B));
    CHECK_CUDA(cudaMalloc(&dSF, sizeof(double) * B));
    CHECK_CUDA(cudaMalloc(&dBA, sizeof(double) * NN * NN));
    CHECK_CUDA(cudaMalloc(&dLIN, sizeof(int64_t) * B));
    CHECK_CUDA(cudaMalloc(&dLF, sizeof(int64_t) * B));
    CHECK_CUDA(cudaMalloc(&dERR, sizeof(double) * B));
    CHECK_CUDA(cudaMalloc(&dRES, sizeof(int64_t) * nzz));

    CHECK_CUDA(cudaMemcpy(dZZ, ZZ, sizeof(double) * nzz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dORIG, ORIG, sizeof(double) * nzz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dACC, ACC, sizeof(double) * B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dSF, SF, sizeof(double) * B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBA, basis2, sizeof(double) * NN * NN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dLIN, LIN, sizeof(int64_t) * B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dRES, 0, sizeof(int64_t) * nzz));

    {
        int threads = 128;
        int grid = (B + threads - 1) / threads;
        quant_kernel<<<grid, threads>>>(dZZ, dORIG, dACC, dSF, dBA, dLIN, dLF, dERR, dRES, B);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemcpy(LF, dLF, sizeof(int64_t) * B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ERR, dERR, sizeof(double) * B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(RES, dRES, sizeof(int64_t) * nzz, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(dZZ);
    cudaFree(dORIG);
    cudaFree(dACC);
    cudaFree(dSF);
    cudaFree(dBA);
    cudaFree(dLIN);
    cudaFree(dLF);
    cudaFree(dERR);
    cudaFree(dRES);
    return rc;
}
