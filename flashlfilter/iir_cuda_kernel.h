#pragma once
#include <torch/extension.h>

void cuda_lfilter_core_loop_chunked(
    const at::Tensor& input, const at::Tensor& a_flipped,
    at::Tensor& output, int64_t chunk_size);