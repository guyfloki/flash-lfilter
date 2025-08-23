#include <torch/extension.h>
#include <vector>
#include "iir_cuda_kernel.h"

namespace idx = torch::indexing;

static at::Tensor fused_cpu_forward(
    const at::Tensor& x,          // [B,C,N] CPU
    const at::Tensor& a_norm,     // [C,K]   CPU
    const at::Tensor& b_norm      // [C,K]   CPU
) {
  TORCH_CHECK(!x.is_cuda() && !a_norm.is_cuda() && !b_norm.is_cuda(), "expected CPU tensors");
  const auto B = x.size(0), C = x.size(1), N = x.size(2), K = a_norm.size(1);
  auto y = torch::empty_like(x);
  for (int64_t b=0;b<B;++b) {
    for (int64_t c=0;c<C;++c) {
      auto xc = x.index({b,c});
      auto yc = y.index({b,c});
      auto a  = a_norm.index({c});
      auto bn = b_norm.index({c});
      std::vector<double> yhist(std::max<int64_t>(K-1,0), 0.0);
      std::vector<double> xring(std::max<int64_t>(K,1),    0.0);
      int pos = 0;
      for (int64_t t=0;t<N;++t) {
        double xnew = xc[t].item<double>();
        if (K>0) xring[pos] = xnew;
        double fir = 0.0;
        for (int64_t j=0;j<K;++j) {
          int idx = (pos - j); if (idx < 0) idx += K;
          fir += bn[j].item<double>() * xring[idx];
        }
        double acc = fir;
        for (int64_t k=1;k<K;++k) acc -= a[k].item<double>() * yhist[k-1];
        if (K>1) {
          for (int64_t k=K-2;k>=1;--k) yhist[k] = yhist[k-1];
          if (K-1>0) yhist[0] = acc;
        }
        yc[t] = acc;
        if (K>0) pos = (pos + 1) % K;
      }
    }
  }
  return y;
}

static at::Tensor lfilter_forward_fused_impl(
    const at::Tensor& waveform,   // [B,C,N]
    const at::Tensor& a_coeffs,   // [C,K]
    const at::Tensor& b_coeffs    // [C,K]
) {
  TORCH_CHECK(waveform.dim()==3, "waveform [B,C,N]");
  TORCH_CHECK(a_coeffs.dim()==2 && b_coeffs.dim()==2, "a,b [C,K]");
  TORCH_CHECK(a_coeffs.size(0)==waveform.size(1) && b_coeffs.size(0)==waveform.size(1), "C mismatch");
  TORCH_CHECK(a_coeffs.size(1)==b_coeffs.size(1), "K mismatch");

  const auto K = b_coeffs.size(1);
  auto a0 = a_coeffs.index({idx::Slice(), 0}).unsqueeze(1);
  constexpr double EPS = 1e-8;
  TORCH_CHECK(a0.abs().min().item<double>()>EPS, "a[:,0] must be non-zero");

  auto a_norm = a_coeffs / a0;
  auto b_norm = b_coeffs / a0;

  if (waveform.is_cuda()) {
    auto y = torch::empty_like(waveform);
    cuda_lfilter_fused_persistent(
        waveform.contiguous(), a_norm.contiguous(), b_norm.contiguous(), y);
    return y;
  } else {
    return fused_cpu_forward(waveform.contiguous(), a_norm.contiguous(), b_norm.contiguous());
  }
}

static at::Tensor iir_only_with_fused(
    const at::Tensor& x,          // [B,C,N]
    const at::Tensor& a_norm      // [C,K] (a_norm[:,0]==1)
) {
  auto B = x.size(0), C = x.size(1), N = x.size(2), K = a_norm.size(1);
  auto b_imp = torch::zeros({C, K}, a_norm.options());
  b_imp.index_put_({idx::Slice(), 0}, 1.0);
  if (x.is_cuda()) {
    auto y = torch::empty_like(x);
    cuda_lfilter_fused_persistent(x.contiguous(), a_norm.contiguous(), b_imp.contiguous(), y);
    return y;
  } else {
    return fused_cpu_forward(x.contiguous(), a_norm.contiguous(), b_imp.contiguous());
  }
}

class FlashLFilterAutogradFused : public torch::autograd::Function<FlashLFilterAutogradFused> {
public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& waveform,
                            const at::Tensor& a_coeffs,
                            const at::Tensor& b_coeffs,
                            int64_t /*chunk_size_unused*/) {
    auto y = lfilter_forward_fused_impl(waveform, a_coeffs, b_coeffs);

    auto a0     = a_coeffs.index({idx::Slice(), 0}).unsqueeze(1);
    auto a_norm = a_coeffs / a0;
    auto b_norm = b_coeffs / a0;
    ctx->save_for_backward({waveform, a_norm, b_norm, y, a0});
    return y;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 const torch::autograd::variable_list& grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto waveform    = saved[0];
    auto a_norm      = saved[1];
    auto b_norm      = saved[2];
    auto forward_out = saved[3];
    auto a0          = saved[4];
    auto grad_output = grad_outputs[0];

    const auto B = waveform.size(0), C = waveform.size(1), N = waveform.size(2), K = a_norm.size(1);

    auto grad_output_flipped = grad_output.flip({2});
    auto grad_fir_out_flipped = iir_only_with_fused(grad_output_flipped, a_norm);
    auto grad_fir_out = grad_fir_out_flipped.flip({2});

    auto b_weight = b_norm.flip({1}).unsqueeze(1);
    auto waveform_padded = at::constant_pad_nd(waveform, {K - 1, 0}).contiguous();
    auto grad_tuple = at::convolution_backward(
        grad_fir_out, waveform_padded, b_weight,
        /*bias_sizes=*/c10::nullopt,
        /*stride=*/{1}, /*padding=*/{0}, /*dilation=*/{1},
        /*transposed=*/false, /*output_padding=*/{0},
        /*groups=*/C,
        /*output_mask=*/{true, true, false}
    );
    auto grad_waveform_padded = std::get<0>(grad_tuple);
    auto grad_waveform = grad_waveform_padded.index({idx::Slice(), idx::Slice(), idx::Slice(K - 1, torch::indexing::None)}).contiguous();
    auto grad_b_weight = std::get<1>(grad_tuple);
    auto grad_b_norm   = grad_b_weight.squeeze(1).flip({1});

    auto conv_input_padded = at::constant_pad_nd(forward_out, {K - 1, 0});
    auto conv_input  = conv_input_padded.view({1, B * C, N + K - 1});
    auto conv_weight = grad_fir_out.view({B * C, 1, N});
    auto grad_a_norm_batched_flipped = -at::convolution(
        conv_input, conv_weight, c10::nullopt,
        {1},{0},{1}, false,{0}, B*C).squeeze(0);
    auto grad_a_norm_batched = grad_a_norm_batched_flipped.flip({1});
    auto grad_a_norm_ = grad_a_norm_batched.view({B, C, K}).sum(0);

    auto grad_a0 = (-(grad_a_norm_ * a_norm).sum(1, true)
                    -(grad_b_norm  * b_norm).sum(1, true)) / a0;

    auto grad_a = grad_a_norm_ / a0;
    grad_a.index_put_({idx::Slice(), 0}, grad_a.index({idx::Slice(), 0}) + grad_a0.squeeze(1));
    auto grad_b = grad_b_norm / a0;

    return {grad_waveform, grad_a, grad_b, torch::Tensor()};
  }
};

static at::Tensor lfilter_forward_fused(const at::Tensor& x, const at::Tensor& a, const at::Tensor& b, int64_t /*chunk*/){
  return lfilter_forward_fused_impl(x, a, b);
}

TORCH_LIBRARY(flashlfilterx, m) {
  m.def("lfilter_forward_fused(Tensor waveform, Tensor a, Tensor b, int chunk_size) -> Tensor");
  m.def("lfilter_autograd_fused(Tensor waveform, Tensor a, Tensor b, int chunk_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(flashlfilterx, CompositeExplicitAutograd, m) {
  m.impl("lfilter_forward_fused",      lfilter_forward_fused);
  m.impl("lfilter_autograd_fused", [](const at::Tensor& x, const at::Tensor& a, const at::Tensor& b, int64_t chunk){
    return FlashLFilterAutogradFused::apply(x, a, b, chunk);
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
