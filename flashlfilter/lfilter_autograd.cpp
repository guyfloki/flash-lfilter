#include <torch/extension.h>
#include <vector>
#include "iir_cuda_kernel.h"

namespace idx = torch::indexing;

static void iir_core_generic_loop(
    const at::Tensor& input,        
    const at::Tensor& a_flipped,    
    at::Tensor& padded              
) {
  TORCH_CHECK(!input.is_cuda(), "iir_core_generic_loop: expected CPU tensor for input");
  TORCH_CHECK(!a_flipped.is_cuda(), "iir_core_generic_loop: expected CPU tensor for a_flipped");
  TORCH_CHECK(!padded.is_cuda(), "iir_core_generic_loop: expected CPU tensor for padded");

  const int64_t B = input.size(0);
  const int64_t C = input.size(1);
  const int64_t N = input.size(2);
  const int64_t K = a_flipped.size(1);

  padded.zero_();

  for (int64_t bc = 0; bc < B * C; ++bc) {
    const int64_t b = bc / C;
    const int64_t c = bc % C;

    auto in  = input.index({b, c});
    auto out = padded.index({b, c});
    auto a_c = a_flipped.index({c});

    for (int64_t t = 0; t < N; ++t) {
      double acc = in[t].item<double>();
      for (int64_t k = 1; k < K; ++k) {
        acc -= a_c[K - 1 - k].item<double>() * out[t + K - 1 - k].item<double>();
      }
      out[t + K - 1] = acc;
    }
  }
}


static at::Tensor run_iir(
    const at::Tensor& input,        // [B, C, N]
    const at::Tensor& a_flipped,    // [C, K]
    int64_t chunk_size
) {
  TORCH_CHECK(input.dim() == 3, "run_iir: input must be [B, C, N]");
  TORCH_CHECK(a_flipped.dim() == 2, "run_iir: a_flipped must be [C, K]");
  TORCH_CHECK(input.size(1) == a_flipped.size(0), "run_iir: C must match between input and a_flipped");

  const auto B = input.size(0);
  const auto C = input.size(1);
  const auto N = input.size(2);
  const auto K = a_flipped.size(1);

  auto padded = torch::empty({B, C, N + K - 1}, input.options());

  if (input.is_cuda()) {
    cuda_lfilter_core_loop_chunked(input, a_flipped.contiguous(), padded, chunk_size);
  } else {
    iir_core_generic_loop(input.contiguous(), a_flipped.contiguous(), padded);
  }

  return padded.index({idx::Slice(), idx::Slice(), idx::Slice(K - 1, torch::indexing::None)});
}


class FlashLFilterAutograd : public torch::autograd::Function<FlashLFilterAutograd> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& waveform,   // [B, C, N]
      const at::Tensor& a_coeffs,   // [C, K]
      const at::Tensor& b_coeffs,   // [C, K]
      int64_t chunk_size) {

    TORCH_CHECK(waveform.dim() == 3, "waveform must be [B, C, N]");
    TORCH_CHECK(a_coeffs.dim() == 2 && b_coeffs.dim() == 2, "a_coeffs and b_coeffs must be [C, K]");
    TORCH_CHECK(a_coeffs.size(0) == waveform.size(1), "a_coeffs[C,K]: C must match waveform.size(1)");
    TORCH_CHECK(b_coeffs.size(0) == waveform.size(1), "b_coeffs[C,K]: C must match waveform.size(1)");
    TORCH_CHECK(a_coeffs.size(1) == b_coeffs.size(1), "a_coeffs and b_coeffs must share K");

    const auto K = b_coeffs.size(1);

    auto a0 = a_coeffs.index({idx::Slice(), 0}).unsqueeze(1); // [C,1]
    constexpr double kA0_EPS = 1e-8; 
    TORCH_CHECK(
        a0.abs().min().item<double>() > kA0_EPS,
        "flashlfilter: a[:,0] must be non-zero; require |a0| > ", kA0_EPS
    );

    auto a_norm = a_coeffs / a0;                         // [C,K]
    auto b_norm = b_coeffs / a0;                         // [C,K]
    auto b_weight = b_norm.flip({1}).unsqueeze(1);       // [C,1,K] 

    auto waveform_padded = at::constant_pad_nd(waveform, {K - 1, 0}); // [B,C,N+K-1]

    auto fir_out = at::convolution(
        waveform_padded,
        b_weight,                   // [C,1,K]
        /*bias=*/c10::nullopt,
        /*stride=*/{1},
        /*padding=*/{0},
        /*dilation=*/{1},
        /*transposed=*/false,
        /*output_padding=*/{0},
        /*groups=*/a_coeffs.size(0) // == C
    ); // -> [B,C,N]

    auto a_flipped = a_norm.flip({1}).contiguous();      // [C,K]
    auto final_out = run_iir(fir_out, a_flipped, chunk_size); // [B,C,N]

    ctx->save_for_backward({waveform, a_norm, b_norm, final_out, a0});
    ctx->saved_data["chunk_size"] = chunk_size;

    return final_out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {

    auto saved = ctx->get_saved_variables();
    TORCH_CHECK(saved.size() == 5, "internal error: expected 5 saved tensors");

    auto waveform    = saved[0];  // [B,C,N]
    auto a_norm      = saved[1];  // [C,K]
    auto b_norm      = saved[2];  // [C,K]
    auto forward_out = saved[3];  // [B,C,N]
    auto a0          = saved[4];  // [C,1] 

    const auto chunk_size = ctx->saved_data["chunk_size"].toInt();
    auto grad_output = grad_outputs[0]; // [B,C,N]

    const auto B = waveform.size(0);
    const auto C = waveform.size(1);
    const auto N = waveform.size(2);
    const auto K = a_norm.size(1);

    auto a_flipped = a_norm.flip({1});                  // [C,K]
    auto grad_output_flipped = grad_output.flip({2});   // [B,C,N]
    auto grad_fir_out_flipped = run_iir(grad_output_flipped, a_flipped, chunk_size); // [B,C,N]
    auto grad_fir_out = grad_fir_out_flipped.flip({2}); // [B,C,N]
    
    auto b_weight = b_norm.flip({1}).unsqueeze(1);      // [C,1,K]
    
    auto waveform_padded = at::constant_pad_nd(waveform, {K - 1, 0}).contiguous(); // [B,C,N+K-1]
    
    auto grad_tuple = at::convolution_backward(
        /*grad_output=*/grad_fir_out,    // [B,C,N]
        /*input=*/waveform_padded,       // [B,C,N+K-1]
        /*weight=*/b_weight,             // [C,1,K]
        /*bias_sizes=*/c10::nullopt,
        /*stride=*/{1},
        /*padding=*/{0},                
        /*dilation=*/{1},
        /*transposed=*/false,
        /*output_padding=*/{0},
        /*groups=*/C,
        /*output_mask=*/{true, true, false}
    );
    
    auto grad_waveform_padded = std::get<0>(grad_tuple);                                    // [B,C,N+K-1]
    auto grad_waveform = grad_waveform_padded.index({idx::Slice(), idx::Slice(), idx::Slice(K - 1, torch::indexing::None)}).contiguous(); // [B,C,N]
    auto grad_b_weight = std::get<1>(grad_tuple);                                           // [C,1,K]
    auto grad_b_norm   = grad_b_weight.squeeze(1).flip({1});                                // [C,K]
    
    auto conv_input_padded = at::constant_pad_nd(forward_out, {K - 1, 0}); // [B,C,N+K-1]
    auto conv_input  = conv_input_padded.view({1, B * C, N + K - 1});      // [1,BC,N+K-1]
    auto conv_weight = grad_fir_out.view({B * C, 1, N});                   // [BC,1,N]
    
    auto grad_a_norm_batched_flipped = -at::convolution(
        conv_input,                // [1,BC,N+K-1]
        conv_weight,               // [BC,1,N]
        /*bias=*/c10::nullopt,
        /*stride=*/{1},
        /*padding=*/{0},
        /*dilation=*/{1},
        /*transposed=*/false,
        /*output_padding=*/{0},
        /*groups=*/B * C
    ).squeeze(0); // [BC,K]
    
    auto grad_a_norm_batched = grad_a_norm_batched_flipped.flip({1}); // [BC,K]
    auto grad_a_norm_ = grad_a_norm_batched.view({B, C, K}).sum(0);   // [C,K]
    
    auto grad_a0 = (-(grad_a_norm_ * a_norm).sum(1, true)
                    -(grad_b_norm  * b_norm).sum(1, true)) / a0;      // [C,1]
    
    auto grad_a = grad_a_norm_ / a0;                                   // [C,K]
    grad_a.index_put_({idx::Slice(), 0},
                      grad_a.index({idx::Slice(), 0}) + grad_a0.squeeze(1));
    auto grad_b = grad_b_norm / a0;                                     // [C,K]
    
    return {grad_waveform, grad_a, grad_b, torch::Tensor()};
  }
};

static at::Tensor lfilter_autograd(
    const at::Tensor& waveform,
    const at::Tensor& a_coeffs,
    const at::Tensor& b_coeffs,
    int64_t chunk_size) {
  return FlashLFilterAutograd::apply(waveform, a_coeffs, b_coeffs, chunk_size);
}

TORCH_LIBRARY(flashlfilter, m) {
  m.def("lfilter_autograd(Tensor waveform, Tensor a, Tensor b, int chunk_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(flashlfilter, Autograd, m) {
  m.impl("lfilter_autograd", lfilter_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
