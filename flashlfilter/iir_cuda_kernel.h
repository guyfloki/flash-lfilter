#pragma once
#include <torch/extension.h>

void cuda_lfilter_fused_persistent(
    const at::Tensor& x,            // [B,C,N], CUDA
    const at::Tensor& a_norm,       // [C,K], CUDA
    const at::Tensor& b_norm,       // [C,K], CUDA
    at::Tensor& y                   // [B,C,N], CUDA 
);
