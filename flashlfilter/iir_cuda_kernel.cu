#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include "iir_cuda_kernel.h"

template <typename scalar_t>
__global__ void zero_prelude_kernel(scalar_t* out, int B, int C, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * C) return;
  scalar_t* out_ptr = out + idx * (N + K - 1);
  for (int i = 0; i < K - 1; ++i) out_ptr[i] = static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void lfilter_chunk_kernel(
    const scalar_t* __restrict__ in, const scalar_t* __restrict__ a_flipped,
    scalar_t* __restrict__ out, int B, int C, int N, int K,
    int t0, int chunk_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * C) return;
  const scalar_t* in_ptr  = in  + idx * N;
  scalar_t*       out_ptr = out + idx * (N + K - 1);
  const scalar_t* a_ptr   = a_flipped + (idx % C) * K;
  for (int t = 0; t < chunk_len; ++t) {
    int g = t0 + t;
    scalar_t acc = in_ptr[g];
    for (int k = 1; k < K; ++k) {
      acc = fmaf(-a_ptr[K - 1 - k], out_ptr[g + K - 1 - k], acc);
    }
    out_ptr[g + K - 1] = acc;
  }
}

void cuda_lfilter_core_loop_chunked(
    const at::Tensor& input, const at::Tensor& a_flipped,
    at::Tensor& output, int64_t chunk_size) {
  const int B = input.size(0), C = input.size(1), N = input.size(2), K = a_flipped.size(1);
  const int total = B * C, threads = 1, blocks = total;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "lfilter_cuda_zero_prelude", [&] {
    zero_prelude_kernel<scalar_t><<<blocks, threads, 0, stream>>>(output.data_ptr<scalar_t>(), B, C, N, K);
  });
  for (int t0 = 0; t0 < N; t0 += chunk_size) {
    int len = std::min<int>(chunk_size, N - t0);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "lfilter_cuda_chunked", [&] {
      lfilter_chunk_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          input.data_ptr<scalar_t>(), a_flipped.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), B, C, N, K, t0, len);
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
}