#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include "iir_cuda_kernel.h"

template <typename T>
__device__ __forceinline__ T FMADD(T a, T b, T c) {
  if constexpr (std::is_same<T,float>::value)       return __fmaf_rn(a, b, c);
  else if constexpr (std::is_same<T,double>::value) return fma(a, b, c);
  else                                              return a * b + c;
}

__host__ __device__ __forceinline__ bool is_pow2_int(int k){ return (k>0) && ((k & (k-1))==0); }

__device__ __forceinline__ float ld_cg_float(const float* p) {
#if __CUDA_ARCH__ >= 700
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
#else
  return *p;
#endif
}
template <typename T>
__device__ __forceinline__ T ld_cg(const T* p) { return *p; }
template <>
__device__ __forceinline__ float ld_cg<float>(const float* p) { return ld_cg_float(p); }

#ifndef FLASHL_MAX_CONST_K
#define FLASHL_MAX_CONST_K 256
#endif
__constant__ float FLASHL_C_A[FLASHL_MAX_CONST_K];
__constant__ float FLASHL_C_B[FLASHL_MAX_CONST_K];

template <typename scalar_t, int MAXK, bool USE_CONST>
__launch_bounds__(1, 1024)
__global__ void fused_fir_iir_kernel_1t(
    const scalar_t* __restrict__ x,   // [B*C, N]
    const scalar_t* __restrict__ a,   // [C, K] (a_norm, a[0]==1)
    const scalar_t* __restrict__ b,   // [C, K] (b_norm)         
    scalar_t* __restrict__ y,         // [B*C, N]
    int B, int C, int N, int K) {
  const int bc = blockIdx.x;
  if (bc >= B*C) return;
  const int c  = bc % C;

  const scalar_t* x_ptr = x + bc * N;
  scalar_t*       y_ptr = y + bc * N;

  scalar_t acoef[MAXK];
  scalar_t bcoef[MAXK];
  if constexpr (!USE_CONST) {
    const scalar_t* a_c = a + c * K;
    const scalar_t* b_c = b + c * K;
    #pragma unroll
    for (int j = 0; j < MAXK; ++j) {
      if (j < K) { acoef[j] = a_c[j]; bcoef[j] = b_c[j]; }
    }
  } else {
    #pragma unroll
    for (int j = 0; j < MAXK; ++j) {
      if (j < K) { acoef[j] = static_cast<scalar_t>(FLASHL_C_A[j]); bcoef[j] = static_cast<scalar_t>(FLASHL_C_B[j]); }
    }
  }

  scalar_t xring[MAXK];
  scalar_t yring[MAXK]; 
  #pragma unroll
  for (int j = 0; j < MAXK; ++j) { xring[j] = scalar_t(0); yring[j] = scalar_t(0); }

  int posx = 0;
  int posy = 0;
  const int L = (K > 0 ? K - 1 : 0);

  const bool POW2K = is_pow2_int(K);
  const bool POW2L = is_pow2_int(L);
  const int KMASK = K - 1;
  const int LMASK = L - 1;

  int t = 0;
  for (; t + 1 < N; t += 2) {
    scalar_t x0 = ld_cg<scalar_t>(x_ptr + t);
    if (K > 0) xring[posx] = x0;

    scalar_t fir0 = 0;
    #pragma unroll
    for (int j = 0; j < MAXK; ++j) {
      if (j >= K) break;
      int idx = posx - j;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      fir0 = FMADD(bcoef[j], xring[idx], fir0);
    }
    scalar_t acc0 = fir0;
    #pragma unroll
    for (int k = 1; k < MAXK; ++k) {
      if (k >= K) break;
      if (L == 0) break;
      int ridx = posy - k;
      if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
      acc0 = FMADD(-acoef[k], yring[ridx], acc0);
    }
    y_ptr[t] = acc0;
    if (L > 0) {
      yring[posy] = acc0;
      ++posy; if (posy == L) posy = 0;
    }
    if (K > 0) { ++posx; if (posx == K) posx = 0; }

    scalar_t x1 = ld_cg<scalar_t>(x_ptr + t + 1);
    if (K > 0) xring[posx] = x1;

    scalar_t fir1 = 0;
    #pragma unroll
    for (int j = 0; j < MAXK; ++j) {
      if (j >= K) break;
      int idx = posx - j;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      fir1 = FMADD(bcoef[j], xring[idx], fir1);
    }
    scalar_t acc1 = fir1;
    #pragma unroll
    for (int k = 1; k < MAXK; ++k) {
      if (k >= K) break;
      if (L == 0) break;
      int ridx = posy - k;
      if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
      acc1 = FMADD(-acoef[k], yring[ridx], acc1);
    }
    y_ptr[t + 1] = acc1;

    if (L > 0) {
      yring[posy] = acc1;
      ++posy; if (posy == L) posy = 0;
    }
    if (K > 0) { ++posx; if (posx == K) posx = 0; }
  }

  if (t < N) {
    scalar_t x0 = ld_cg<scalar_t>(x_ptr + t);
    if (K > 0) xring[posx] = x0;

    scalar_t fir0 = 0;
    #pragma unroll
    for (int j = 0; j < MAXK; ++j) {
      if (j >= K) break;
      int idx = posx - j;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      fir0 = FMADD(bcoef[j], xring[idx], fir0);
    }
    scalar_t acc0 = fir0;
    #pragma unroll
    for (int k = 1; k < MAXK; ++k) {
      if (k >= K) break;
      if (L == 0) break;
      int ridx = posy - k;
      if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
      acc0 = FMADD(-acoef[k], yring[ridx], acc0);
    }
    y_ptr[t] = acc0;
  }
}

template <typename scalar_t, int K, bool USE_CONST>
__launch_bounds__(32, 8)
__global__ void fused_fir_iir_kernel_subwarp8x4(
    const scalar_t* __restrict__ x,   // [B*C, N]
    const scalar_t* __restrict__ a,   // [C, K]
    const scalar_t* __restrict__ b,   // [C, K]
    scalar_t* __restrict__ y,         // [B*C, N]
    int B, int C, int N) {
  static_assert(K>=1 && K<=8, "K must be 1..8 for subwarp8x4");
  const int lane  = threadIdx.x & 31;
  const int group = lane >> 3;               // 0..3
  const int sub   = lane & 7;                // 0..7
  const unsigned mask_full = 0xffffffffu;
  const unsigned mask8 = 0xFFu << (group * 8);

  const int base = blockIdx.x * 4;
  const int bc_g = base + group;
  if (bc_g >= B * C) return;

  const int c_g = bc_g % C;
  const scalar_t* x_ptr = x + bc_g * N;
  scalar_t*       y_ptr = y + bc_g * N;

  scalar_t b_reg = 0;
  scalar_t a_reg = 0;
  if constexpr (USE_CONST) {
    if (sub < K)        b_reg = static_cast<scalar_t>(FLASHL_C_B[sub]);
    if (sub + 1 < K)    a_reg = static_cast<scalar_t>(FLASHL_C_A[sub + 1]);
  } else {
    const scalar_t* a_c   = a + c_g * K;
    const scalar_t* b_c   = b + c_g * K;
    if (sub < K)        b_reg = b_c[sub];
    if (sub + 1 < K)    a_reg = a_c[sub + 1];
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* sm = reinterpret_cast<scalar_t*>(smem_raw);

  constexpr int L    = (K > 0 ? K - 1 : 0);
  constexpr bool POW2K = ((K & (K-1))==0);
  constexpr bool POW2L = (L>0) && ((L & (L-1))==0);
  constexpr int Kpad = (K & 31) == 0 ? (K + 1) : K;
  constexpr int Lpad = (L & 31) == 0 ? (L + 1) : L;
  constexpr int stride = (Kpad + Lpad);

  scalar_t* xring = sm + group * stride;
  scalar_t* yring = xring + Kpad;

  for (int i = sub; i < K; i += 8) xring[i] = scalar_t(0);
  for (int i = sub; i < L; i += 8) yring[i] = scalar_t(0);
  __syncwarp(mask_full);

  int curx = 0, cury = 0;
  constexpr int KMASK = K - 1;
  constexpr int LMASK = (L>0?L-1:0);

  int t = 0;
  for (; t + 1 < N; t += 2) {
    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask8);

    scalar_t part_fir = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir += __shfl_down_sync(mask8, part_fir, off);

    scalar_t part_iir = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir += __shfl_down_sync(mask8, part_iir, off);
    }

    if (sub == 0) {
      scalar_t y0 = part_fir - part_iir;
      y_ptr[t] = y0;
      if (L > 0) { yring[cury] = y0; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
    }
    curx = __shfl_sync(mask8, curx, 0);
    cury = __shfl_sync(mask8, cury, 0);
    __syncwarp(mask8);

    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t + 1);
    __syncwarp(mask8);

    scalar_t part_fir1 = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir1 = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir1 += __shfl_down_sync(mask8, part_fir1, off);

    scalar_t part_iir1 = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir1 = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir1 += __shfl_down_sync(mask8, part_iir1, off);
    }

    if (sub == 0) {
      scalar_t y1 = part_fir1 - part_iir1;
      y_ptr[t + 1] = y1;
      if (L > 0) { yring[cury] = y1; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
    }
    curx = __shfl_sync(mask8, curx, 0);
    cury = __shfl_sync(mask8, cury, 0);
    __syncwarp(mask8);
  }

  if (t < N) {
    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask8);

    scalar_t part_fir = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir += __shfl_down_sync(mask8, part_fir, off);

    scalar_t part_iir = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir += __shfl_down_sync(mask8, part_iir, off);
    }
    if (sub == 0) {
      scalar_t y0 = part_fir - part_iir;
      y_ptr[t] = y0;
    }
  }
}

template <typename scalar_t, bool USE_CONST>
__launch_bounds__(32, 8)
__global__ void fused_fir_iir_kernel_subwarp8x4_dyn(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ y,
    int B, int C, int N, int K) {
  const int lane  = threadIdx.x & 31;
  const int group = lane >> 3;
  const int sub   = lane & 7;
  const unsigned mask_full = 0xffffffffu;
  const unsigned mask8 = 0xFFu << (group * 8);

  const int base = blockIdx.x * 4;
  const int bc_g = base + group;
  if (bc_g >= B * C) return;

  const int c_g = bc_g % C;
  const scalar_t* x_ptr = x + bc_g * N;
  scalar_t*       y_ptr = y + bc_g * N;

  const int L = (K>0?K-1:0);
  const bool POW2K = is_pow2_int(K);
  const bool POW2L = is_pow2_int(L);
  const int KMASK = K-1;
  const int LMASK = (L>0?L-1:0);

  scalar_t b_reg = 0, a_reg = 0;
  if constexpr (USE_CONST) {
    if (sub < K)     b_reg = static_cast<scalar_t>(FLASHL_C_B[sub]);
    if (sub + 1 < K) a_reg = static_cast<scalar_t>(FLASHL_C_A[sub + 1]);
  } else {
    const scalar_t* a_c   = a + c_g * K;
    const scalar_t* b_c   = b + c_g * K;
    if (sub < K)     b_reg = b_c[sub];
    if (sub + 1 < K) a_reg = a_c[sub + 1];
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* sm = reinterpret_cast<scalar_t*>(smem_raw);

  const int Kpad = K + ((K & 31) == 0 ? 1 : 0);
  const int Lpad = L + ((L & 31) == 0 ? 1 : 0);
  const int stride = (Kpad + Lpad);

  scalar_t* xring = sm + group * stride;
  scalar_t* yring = xring + Kpad;

  for (int i = sub; i < K; i += 8) xring[i] = scalar_t(0);
  for (int i = sub; i < L; i += 8) yring[i] = scalar_t(0);
  __syncwarp(mask_full);

  int curx = 0, cury = 0;

  int t = 0;
  for (; t + 1 < N; t += 2) {
    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask8);

    scalar_t part_fir = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir += __shfl_down_sync(mask8, part_fir, off);

    scalar_t part_iir = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir += __shfl_down_sync(mask8, part_iir, off);
    }

    if (sub == 0) {
      scalar_t y0 = part_fir - part_iir;
      y_ptr[t] = y0;
      if (L > 0) { yring[cury] = y0; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
    }
    curx = __shfl_sync(mask8, curx, 0);
    cury = __shfl_sync(mask8, cury, 0);
    __syncwarp(mask8);

    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t + 1);
    __syncwarp(mask8);

    scalar_t part_fir1 = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir1 = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir1 += __shfl_down_sync(mask8, part_fir1, off);

    scalar_t part_iir1 = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir1 = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir1 += __shfl_down_sync(mask8, part_iir1, off);
    }
    if (sub == 0) {
      scalar_t y1 = part_fir1 - part_iir1;
      y_ptr[t + 1] = y1;
      if (L > 0) { yring[cury] = y1; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
    }
    curx = __shfl_sync(mask8, curx, 0);
    cury = __shfl_sync(mask8, cury, 0);
    __syncwarp(mask8);
  }

  if (t < N) {
    if (sub == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask8);

    scalar_t part_fir = 0;
    if (sub < K) {
      int idx = curx - sub;
      if (POW2K) idx &= KMASK; else idx += (idx >> 31) & K;
      part_fir = b_reg * xring[idx];
    }
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) part_fir += __shfl_down_sync(mask8, part_fir, off);

    scalar_t part_iir = 0;
    if (L > 0) {
      int k = sub + 1;
      if (k < K) {
        int ridx = cury - k;
        if (POW2L) ridx &= LMASK; else ridx += (ridx >> 31) & L;
        part_iir = a_reg * yring[ridx];
      }
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) part_iir += __shfl_down_sync(mask8, part_iir, off);
    }
    if (sub == 0) {
      scalar_t y0 = part_fir - part_iir;
      y_ptr[t] = y0;
    }
  }
}

template <typename scalar_t, int K, bool USE_CONST>
__launch_bounds__(32, 4)
__global__ void fused_fir_iir_kernel_warp32(
    const scalar_t* __restrict__ x,   // [B*C, N]
    const scalar_t* __restrict__ a,   // [C, K]
    const scalar_t* __restrict__ b,   // [C, K]
    scalar_t* __restrict__ y,         // [B*C, N]
    int B, int C, int N) {
  static_assert(K>=1 && K<=64, "K must be 1..64 for warp32");
  const int bc = blockIdx.x;
  if (bc >= B*C) return;
  const int c    = bc % C;
  const int lane = threadIdx.x & 31;
  const unsigned mask = 0xffffffffu;

  const scalar_t* x_ptr = x + bc * N;
  scalar_t*       y_ptr = y + bc * N;

  constexpr int L    = (K > 0 ? K - 1 : 0);
  constexpr bool POW2K = ((K & (K-1))==0);
  constexpr bool POW2L = (L>0) && ((L & (L-1))==0);
  constexpr int KMASK = K - 1;
  constexpr int LMASK = (L>0?L-1:0);

  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* xring = reinterpret_cast<scalar_t*>(smem_raw);

  constexpr int Kpad = (K & 31) == 0 ? (K + 1) : K;
  constexpr int Lpad = (L & 31) == 0 ? (L + 1) : L;

  scalar_t* yring = xring + Kpad;

  for (int i = lane; i < K; i += 32) xring[i] = scalar_t(0);
  for (int i = lane; i < L; i += 32) yring[i] = scalar_t(0);
  __syncwarp(mask);

  scalar_t b0 = 0, b1 = 0, a0 = 0, a1 = 0;
  if constexpr (USE_CONST) {
    if (lane < K)        b0 = static_cast<scalar_t>(FLASHL_C_B[lane]);
    if (lane + 32 < K)   b1 = static_cast<scalar_t>(FLASHL_C_B[lane + 32]);
    if (lane + 1 < K)    a0 = static_cast<scalar_t>(FLASHL_C_A[lane + 1]);
    if (lane + 33 < K)   a1 = static_cast<scalar_t>(FLASHL_C_A[lane + 33]);
  } else {
    const scalar_t* a_c   = a + c * K;
    const scalar_t* b_c   = b + c * K;
    if (lane < K)        b0 = b_c[lane];
    if (lane + 32 < K)   b1 = b_c[lane + 32];
    if (lane + 1 < K)    a0 = a_c[lane + 1];
    if (lane + 33 < K)   a1 = a_c[lane + 33];
  }

  int posx = 0, posy = 0;

  int t = 0;
  for (; t + 1 < N; t += 2) {
    int curx = __shfl_sync(mask, posx, 0);
    int cury = __shfl_sync(mask, posy, 0);

    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask);

    scalar_t sum_fir = 0;
    {
      int idx1 = curx - lane;  if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
      sum_fir = FMADD(b0, xring[idx1], sum_fir);
      if (lane + 32 < K) {
        int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
        sum_fir = FMADD(b1, xring[idx2], sum_fir);
      }
    }

    scalar_t sum_iir = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir = FMADD(a0, yring[ridx1], sum_iir); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir = FMADD(a1, yring[ridx2], sum_iir); }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_fir += __shfl_down_sync(mask, sum_fir, off);
      sum_iir += __shfl_down_sync(mask, sum_iir, off);
    }

    if (lane == 0) {
      scalar_t yval = sum_fir - sum_iir;
      y_ptr[t] = yval;
      if (L > 0) { yring[cury] = yval; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
      posx = curx; posy = cury;
    }
    __syncwarp(mask);

    curx = __shfl_sync(mask, posx, 0);
    cury = __shfl_sync(mask, posy, 0);

    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t + 1);
    __syncwarp(mask);

    scalar_t sum_fir1 = 0;
    {
      int idx1 = curx - lane;  if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
      sum_fir1 = FMADD(b0, xring[idx1], sum_fir1);
      if (lane + 32 < K) {
        int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
        sum_fir1 = FMADD(b1, xring[idx2], sum_fir1);
      }
    }

    scalar_t sum_iir1 = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir1 = FMADD(a0, yring[ridx1], sum_iir1); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir1 = FMADD(a1, yring[ridx2], sum_iir1); }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_fir1 += __shfl_down_sync(mask, sum_fir1, off);
      sum_iir1 += __shfl_down_sync(mask, sum_iir1, off);
    }

    if (lane == 0) {
      scalar_t yval = sum_fir1 - sum_iir1;
      y_ptr[t + 1] = yval;
      if (L > 0) { yring[cury] = yval; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
      posx = curx; posy = cury;
    }
    __syncwarp(mask);
  }

  if (t < N) {
    int curx = __shfl_sync(mask, posx, 0);
    int cury = __shfl_sync(mask, posy, 0);
    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask);

    scalar_t sum_fir = 0;
    {
      int idx1 = curx - lane;  if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
      sum_fir = FMADD(b0, xring[idx1], sum_fir);
      if (lane + 32 < K) {
        int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
        sum_fir = FMADD(b1, xring[idx2], sum_fir);
      }
    }
    scalar_t sum_iir = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir = FMADD(a0, yring[ridx1], sum_iir); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir = FMADD(a1, yring[ridx2], sum_iir); }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_fir += __shfl_down_sync(mask, sum_fir, off);
      sum_iir += __shfl_down_sync(mask, sum_iir, off);
    }
    if (lane == 0) y_ptr[t] = sum_fir - sum_iir;
  }
}

template <typename scalar_t, bool USE_CONST>
__launch_bounds__(32, 4)
__global__ void fused_fir_iir_kernel_warp32_dyn(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ y,
    int B, int C, int N, int K) {
  const int bc = blockIdx.x;
  if (bc >= B*C) return;
  const int c    = bc % C;
  const int lane = threadIdx.x & 31;
  const unsigned mask = 0xffffffffu;

  const scalar_t* x_ptr = x + bc * N;
  scalar_t*       y_ptr = y + bc * N;

  const int L = (K>0?K-1:0);
  const bool POW2K = is_pow2_int(K);
  const bool POW2L = is_pow2_int(L);
  const int KMASK = K-1;
  const int LMASK = (L>0?L-1:0);

  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* xring = reinterpret_cast<scalar_t*>(smem_raw);

  const int Kpad = K + ((K & 31) == 0 ? 1 : 0);
  const int Lpad = L + ((L & 31) == 0 ? 1 : 0);
  scalar_t* yring = xring + Kpad;

  for (int i = lane; i < K; i += 32) xring[i] = scalar_t(0);
  for (int i = lane; i < L; i += 32) yring[i] = scalar_t(0);
  __syncwarp(mask);

  scalar_t b0 = 0, b1 = 0, a0 = 0, a1 = 0;
  if constexpr (USE_CONST) {
    if (lane < K)        b0 = static_cast<scalar_t>(FLASHL_C_B[lane]);
    if (lane + 32 < K)   b1 = static_cast<scalar_t>(FLASHL_C_B[lane + 32]);
    if (lane + 1 < K)    a0 = static_cast<scalar_t>(FLASHL_C_A[lane + 1]);
    if (lane + 33 < K)   a1 = static_cast<scalar_t>(FLASHL_C_A[lane + 33]);
  } else {
    const scalar_t* a_c   = a + c * K;
    const scalar_t* b_c   = b + c * K;
    if (lane < K)        b0 = b_c[lane];
    if (lane + 32 < K)   b1 = b_c[lane + 32];
    if (lane + 1 < K)    a0 = a_c[lane + 1];
    if (lane + 33 < K)   a1 = a_c[lane + 33];
  }

  int posx = 0, posy = 0;

  int t = 0;
  for (; t + 1 < N; t += 2) {
    int curx = __shfl_sync(mask, posx, 0);
    int cury = __shfl_sync(mask, posy, 0);
    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask);

    scalar_t sum_fir = 0;
    int idx1 = curx - lane; if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
    sum_fir = FMADD(b0, xring[idx1], sum_fir);
    if (lane + 32 < K) {
      int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
      sum_fir = FMADD(b1, xring[idx2], sum_fir);
    }

    scalar_t sum_iir = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir = FMADD(a0, yring[ridx1], sum_iir); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir = FMADD(a1, yring[ridx2], sum_iir); }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { sum_fir += __shfl_down_sync(mask, sum_fir, off); sum_iir += __shfl_down_sync(mask, sum_iir, off); }

    if (lane == 0) {
      scalar_t yval = sum_fir - sum_iir;
      y_ptr[t] = yval;
      if (L > 0) { yring[cury] = yval; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
      posx = curx; posy = cury;
    }
    __syncwarp(mask);

    curx = __shfl_sync(mask, posx, 0);
    cury = __shfl_sync(mask, posy, 0);
    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t + 1);
    __syncwarp(mask);

    scalar_t sum_fir1 = 0;
    idx1 = curx - lane; if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
    sum_fir1 = FMADD(b0, xring[idx1], sum_fir1);
    if (lane + 32 < K) {
      int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
      sum_fir1 = FMADD(b1, xring[idx2], sum_fir1);
    }

    scalar_t sum_iir1 = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir1 = FMADD(a0, yring[ridx1], sum_iir1); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir1 = FMADD(a1, yring[ridx2], sum_iir1); }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { sum_fir1 += __shfl_down_sync(mask, sum_fir1, off); sum_iir1 += __shfl_down_sync(mask, sum_iir1, off); }

    if (lane == 0) {
      scalar_t yval = sum_fir1 - sum_iir1;
      y_ptr[t + 1] = yval;
      if (L > 0) { yring[cury] = yval; ++cury; if (cury == L) cury = 0; }
      if (K > 0) { ++curx; if (curx == K) curx = 0; }
      posx = curx; posy = cury;
    }
    __syncwarp(mask);
  }

  if (t < N) {
    int curx = __shfl_sync(mask, posx, 0);
    int cury = __shfl_sync(mask, posy, 0);
    if (lane == 0) xring[curx] = ld_cg<scalar_t>(x_ptr + t);
    __syncwarp(mask);

    scalar_t sum_fir = 0;
    int idx1 = curx - lane; if (POW2K) idx1 &= KMASK; else idx1 += (idx1 >> 31) & K;
    sum_fir = FMADD(b0, xring[idx1], sum_fir);
    if (lane + 32 < K) {
      int idx2 = curx - (lane + 32); if (POW2K) idx2 &= KMASK; else idx2 += (idx2 >> 31) & K;
      sum_fir = FMADD(b1, xring[idx2], sum_fir);
    }

    scalar_t sum_iir = 0;
    if (L > 0) {
      if (lane + 1 < K)  { int ridx1 = cury - (lane + 1);  if (POW2L) ridx1 &= LMASK; else ridx1 += (ridx1 >> 31) & L; sum_iir = FMADD(a0, yring[ridx1], sum_iir); }
      if (lane + 33 < K) { int ridx2 = cury - (lane + 33); if (POW2L) ridx2 &= LMASK; else ridx2 += (ridx2 >> 31) & L; sum_iir = FMADD(a1, yring[ridx2], sum_iir); }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_fir += __shfl_down_sync(mask, sum_fir, off);
      sum_iir += __shfl_down_sync(mask, sum_iir, off);
    }
    if (lane == 0) y_ptr[t] = sum_fir - sum_iir;
  }
}

static bool rows_all_equal_float(const at::Tensor& t) {
  if (t.size(0) <= 1) return true;
  auto first = t.index({0}).contiguous();
  auto expanded = first.unsqueeze(0).expand_as(t);
  return t.equal(expanded);
}

template <typename scalar_t>
static void launch_fused_kernel_impl(
    const at::Tensor& x,
    const at::Tensor& a_norm,
    const at::Tensor& b_norm,
    at::Tensor& y,
    int B, int C, int N, int K,
    bool use_const_coeffs) {

  const int BC = B * C;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (K <= 8) {
    const int blocks = (BC + 4 - 1) / 4;
    const int L    = (K > 0 ? K - 1 : 0);
    const int Kpad = K + ((K & 31) == 0 ? 1 : 0);
    const int Lpad = L + ((L & 31) == 0 ? 1 : 0);
    const size_t per_group = static_cast<size_t>(Kpad + Lpad);
    const size_t sh_elems  = per_group * 4;
    const size_t sh_bytes  = sh_elems * sizeof(scalar_t);
    dim3 grid(blocks), cta(32);

    switch (K) {
      case 1: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,1,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,1,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 2: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,2,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,2,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 3: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,3,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,3,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 4: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,4,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,4,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 5: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,5,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,5,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 6: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,6,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,6,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 7: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,7,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,7,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 8: if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4<scalar_t,8,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
              else                   fused_fir_iir_kernel_subwarp8x4<scalar_t,8,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      default:
        if (use_const_coeffs) fused_fir_iir_kernel_subwarp8x4_dyn<scalar_t,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N, K);
        else                  fused_fir_iir_kernel_subwarp8x4_dyn<scalar_t,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    }
  }
  else if (K <= 64) {
    const int blocks = BC;
    const int L    = (K > 0 ? K - 1 : 0);
    const int Kpad = K + ((K & 31) == 0 ? 1 : 0);
    const int Lpad = L + ((L & 31) == 0 ? 1 : 0);
    size_t sh_elems = static_cast<size_t>(Kpad) + static_cast<size_t>(Lpad);
    size_t sh_bytes = sh_elems * sizeof(scalar_t);
    dim3 grid(blocks), cta(32);

    switch (K) {
      case 9:   if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,9 ,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,9 ,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 10:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,10,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,10,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 12:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,12,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,12,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 16:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,16,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,16,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 32:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,32,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,32,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 48:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,48,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,48,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      case 64:  if (use_const_coeffs) fused_fir_iir_kernel_warp32<scalar_t,64,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N);
                else                   fused_fir_iir_kernel_warp32<scalar_t,64,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N); break;
      default:
        if (use_const_coeffs) fused_fir_iir_kernel_warp32_dyn<scalar_t,true ><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, y.data_ptr<scalar_t>(), B, C, N, K);
        else                  fused_fir_iir_kernel_warp32_dyn<scalar_t,false><<<grid, cta, sh_bytes, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    }
  }
  else if (K <= 256) {
    const int blocks = B * C;
    dim3 grid(blocks), cta(1);
         if (K <=   8) fused_fir_iir_kernel_1t<scalar_t,  8,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else if (K <=  16) fused_fir_iir_kernel_1t<scalar_t, 16,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else if (K <=  32) fused_fir_iir_kernel_1t<scalar_t, 32,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else if (K <=  64) fused_fir_iir_kernel_1t<scalar_t, 64,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else if (K <=  96) fused_fir_iir_kernel_1t<scalar_t, 96,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else if (K <= 128) fused_fir_iir_kernel_1t<scalar_t,128,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
    else               fused_fir_iir_kernel_1t<scalar_t,256,false><<<grid, cta, 0, stream>>>(x.data_ptr<scalar_t>(), a_norm.data_ptr<scalar_t>(), b_norm.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, N, K);
  } else {
    AT_ERROR("fused kernel supports K up to 256 (got K=", K, ")");
  }

  AT_CUDA_CHECK(cudaGetLastError());
}


void cuda_lfilter_fused_persistent(
    const at::Tensor& x,
    const at::Tensor& a_norm,
    const at::Tensor& b_norm,
    at::Tensor& y) {
  const int B = x.size(0);
  const int C = x.size(1);
  const int N = x.size(2);
  const int K = a_norm.size(1);

  bool use_const = false;
  if (x.scalar_type() == at::kFloat && K <= FLASHL_MAX_CONST_K && C >= 1) {
    auto a0h = a_norm.to(at::kCPU);
    auto b0h = b_norm.to(at::kCPU);
    auto rows_all_equal_float = [](const at::Tensor& t)->bool{
      if (t.size(0) <= 1) return true;
      auto f = t.index({0}).contiguous();
      return t.equal(f.unsqueeze(0).expand_as(t));
    };
    if (rows_all_equal_float(a0h) && rows_all_equal_float(b0h)) {
      auto ar0 = a0h.index({0}).contiguous();
      auto br0 = b0h.index({0}).contiguous();
      AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(FLASHL_C_A, ar0.data_ptr<float>(), sizeof(float)*K, 0, cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream()));
      AT_CUDA_CHECK(cudaMemcpyToSymbolAsync(FLASHL_C_B, br0.data_ptr<float>(), sizeof(float)*K, 0, cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream()));
      use_const = true;
    }
  }

  if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
    auto xf = x.to(at::kFloat).contiguous();
    auto af = a_norm.to(at::kFloat).contiguous();
    auto bf = b_norm.to(at::kFloat).contiguous();
    auto yf = at::empty_like(xf);
    launch_fused_kernel_impl<float>(xf, af, bf, yf, B, C, N, K, use_const);
    y.copy_(yf.to(x.scalar_type()));
    return;
  }

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_fir_iir", [&]{
    launch_fused_kernel_impl<scalar_t>(x.contiguous(), a_norm.contiguous(), b_norm.contiguous(), y, B, C, N, K, use_const);
  });
}
