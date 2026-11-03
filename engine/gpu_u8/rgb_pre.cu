// cm=6 pre-verification error on the GPU: planes from the quantised coefficients (basis sum),
// clip(+128), YCbCr -> RGB exactly as engine/blocks.ycbcr_to_rgb (same operation order, round
// half to even), max |orig - rgb| over the block, and a flag for pixels whose pre-rounding value
// lies within 1e-9 of a half-integer (the rounding decision is then not robust: the caller
// recomputes those rows exactly). Build with -fmad=false.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef BSZ
#define BSZ 8
#endif
#define NN (BSZ * BSZ)
#define TIE_TOL 1e-9

__device__ __forceinline__ double clip255(double v) {
    if (v < 0.0) return 0.0;
    if (v > 255.0) return 255.0;
    return v;
}

__device__ __forceinline__ double clip_u8(double v) {
    return clip255(v);
}

__device__ __forceinline__ double rnd(double x) {
    // np.round(x) for float64: round half to even
    double r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2.0) != 0.0) r -= 1.0;
    return r;
}

__device__ __forceinline__ bool is_tie(double x) {
    double f = x - floor(x);
    return fabs(f - 0.5) < TIE_TOL;
}

// plane[p] = clip(sum_k res[k]/sf * basis2[k,p] + 128), k ascending
__device__ void plane_from_res(const int64_t* res, double sf, const double* basis2, double* plane) {
    double cdq[NN];
    for (int k = 0; k < NN; ++k) cdq[k] = (double)res[k] / sf;
    for (int p = 0; p < NN; ++p) {
        double rec = 0.0;
        for (int k = 0; k < NN; ++k) rec += cdq[k] * basis2[k * NN + p];
        plane[p] = clip_u8(rec + 128.0);
    }
}

__global__ void rgb_pre_kernel(const int64_t* __restrict__ RY, const int64_t* __restrict__ RCB,
                               const int64_t* __restrict__ RCR, const float* __restrict__ RF,
                               const float* __restrict__ GF, const float* __restrict__ BF,
                               const double* __restrict__ basis2, double sf, int B,
                               double* __restrict__ err, int32_t* __restrict__ flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    double Y[NN], Cb[NN], Cr[NN];
    plane_from_res(RY + (size_t)i * NN, sf, basis2, Y);
    plane_from_res(RCB + (size_t)i * NN, sf, basis2, Cb);
    plane_from_res(RCR + (size_t)i * NN, sf, basis2, Cr);

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
        double dr = fabs((double)RF[(size_t)i * NN + p] - R);
        double dg = fabs((double)GF[(size_t)i * NN + p] - G);
        double db = fabs((double)BF[(size_t)i * NN + p] - Bc);
        if (dr > mx) mx = dr;
        if (dg > mx) mx = dg;
        if (db > mx) mx = db;
    }
    err[i] = mx;
    flag[i] = tie;
}

#define CHECK_CUDA(call)                                  \
    do {                                                  \
        cudaError_t _e = (call);                          \
        if (_e != cudaSuccess) {                          \
            rc = (int)_e;                                 \
            goto cleanup;                                 \
        }                                                 \
    } while (0)

extern "C" int rgb_pre_gpu(const int64_t* RY, const int64_t* RCB, const int64_t* RCR,
                           const float* RF, const float* GF, const float* BF,
                           const double* basis2, double sf, int B, double* err, int32_t* flag) {
    int rc = 0;
    int64_t *dRY = NULL, *dRCB = NULL, *dRCR = NULL;
    float *dRF = NULL, *dGF = NULL, *dBF = NULL;
    double *dBA = NULL, *dERR = NULL;
    int32_t* dFLAG = NULL;
    size_t nn = (size_t)B * NN;

    CHECK_CUDA(cudaMalloc(&dRY, sizeof(int64_t) * nn));
    CHECK_CUDA(cudaMalloc(&dRCB, sizeof(int64_t) * nn));
    CHECK_CUDA(cudaMalloc(&dRCR, sizeof(int64_t) * nn));
    CHECK_CUDA(cudaMalloc(&dRF, sizeof(float) * nn));
    CHECK_CUDA(cudaMalloc(&dGF, sizeof(float) * nn));
    CHECK_CUDA(cudaMalloc(&dBF, sizeof(float) * nn));
    CHECK_CUDA(cudaMalloc(&dBA, sizeof(double) * NN * NN));
    CHECK_CUDA(cudaMalloc(&dERR, sizeof(double) * B));
    CHECK_CUDA(cudaMalloc(&dFLAG, sizeof(int32_t) * B));

    CHECK_CUDA(cudaMemcpy(dRY, RY, sizeof(int64_t) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCB, RCB, sizeof(int64_t) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCR, RCR, sizeof(int64_t) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRF, RF, sizeof(float) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dGF, GF, sizeof(float) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBF, BF, sizeof(float) * nn, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBA, basis2, sizeof(double) * NN * NN, cudaMemcpyHostToDevice));

    {
        int threads = 128;
        int grid = (B + threads - 1) / threads;
        rgb_pre_kernel<<<grid, threads>>>(dRY, dRCB, dRCR, dRF, dGF, dBF, dBA, sf, B, dERR, dFLAG);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemcpy(err, dERR, sizeof(double) * B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(flag, dFLAG, sizeof(int32_t) * B, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(dRY);
    cudaFree(dRCB);
    cudaFree(dRCR);
    cudaFree(dRF);
    cudaFree(dGF);
    cudaFree(dBF);
    cudaFree(dBA);
    cudaFree(dERR);
    cudaFree(dFLAG);
    return rc;
}
