# flash-lfilter

Fast per-channel IIR filtering for PyTorch with CUDA and full autograd support.
Works like `lfilter`: a depthwise FIR stage followed by a causal IIR recursion.

**Performance highlight:** on a Tesla T4, this implementation is typically **~10× faster** than `torchaudio.functional.lfilter` for large signals and multiple channels (see Benchmark section).

## Features

* PyTorch custom op (`flashlfilter.ops`) with autograd
* Batch-first tensors: `[B, C, N]`
* CUDA kernel with chunked time processing; CPU fallback included
* Stable parameterization via normalization by `a[:,0]`
* Input validation: requires `|a[:,0]| > 1e-8` and uses the original `a0` in backward for correct gradients

## Requirements

* Python 3.8+
* PyTorch (install the wheel that matches your CUDA)
* CUDA toolkit compatible with your local PyTorch (for building the extension)
* Linux (tested). Other platforms may work but are not officially supported.

## Installation

First install PyTorch for your CUDA version (example for CUDA 12.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install from GitHub:

```bash
pip install --no-build-isolation "git+https://github.com/guyfloki/flash-lfilter.git@main"
```

Editable (local) install for development:

```bash
pip install -U ninja
pip install -e . -v
```

Why `--no-build-isolation`? It lets the build see the already-installed `torch`, which is required by `CUDAExtension`.

## Quick start

```python
import torch
import flashlfilter

B, C, N, K = 2, 3, 16000, 5
x = torch.randn(B, C, N, device="cuda", dtype=torch.float32)

a = torch.zeros(C, K, device=x.device, dtype=torch.float32, requires_grad=True)
b = torch.rand(C, K, device=x.device, dtype=torch.float32, requires_grad=True)
with torch.no_grad():
    a[:, 0] = 1.0 

y = flashlfilter.lfilter_autograd(x, a, b, chunk_size=1024)
loss = y.pow(2).mean()
loss.backward() 
print(a.grad.shape, b.grad.shape)
```

You can also call the op directly:

```python
torch.ops.flashlfilter.lfilter_autograd(x, a, b, 1024)
```

## API

```python
flashlfilter.lfilter_autograd(
    waveform: Tensor,  # [B, C, N]
    a: Tensor,         # [C, K], IIR denominator per channel
    b: Tensor,         # [C, K], FIR numerator per channel
    chunk_size: int    # time chunk length for the CUDA kernel
) -> Tensor           # [B, C, N]
```

Shapes:

* `waveform`: float32/float64, `B x C x N`
* `a`, `b`: `C x K` (one coefficient set per channel)

Behavior:

* Internally normalizes by `a0 = a[:,0:1]` so the effective recursion uses `a_norm` with `a_norm[:,0] == 1`.
* Guard: `|a[:,0]|` must be > `1e-12`; otherwise the op raises an error.
* Autograd saves the original `a0` to produce correct gradients w\.r.t. the unnormalized `a` and `b`.

## Performance benchmark (Tesla T4)

In our tests on a Tesla T4 (16 GB) with CUDA 12.5, PyTorch 2.x, and torchaudio 2.x, this implementation is typically **~10× faster** than `torchaudio.functional.lfilter` on large inputs and multiple channels.

Example benchmark script you can run:

```python
import torch, time, torchaudio, flashlfilter

torch.manual_seed(0)
device = "cuda"
dtype = torch.float32

B, C, N, K = 4, 16, 1_000_000, 8
x = torch.randn(B, C, N, device=device, dtype=dtype)
a = torch.zeros(C, K, device=device, dtype=dtype); a[:, 0] = 1.0
b = torch.randn(C, K, device=device, dtype=dtype) * 0.1

def bench_sync(fn, iters=10, warmup=3):
    # warmup
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
    # timed
    t0 = time.time()
    for _ in range(iters):
        y = fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / iters

def run_flash():
    return flashlfilter.lfilter_autograd(x, a, b, chunk_size=2048)

def run_torchaudio():
    y = torch.empty_like(x)
    for c in range(C):
        y[:, c] = torchaudio.functional.lfilter(
            x[:, c], b[c], a[c]
        )
    return y

t_flash = bench_sync(run_flash)
t_torch = bench_sync(run_torchaudio)

print(f"flash-lfilter: {t_flash*1e3:.1f} ms/iter")
print(f"torchaudio   : {t_torch*1e3:.1f} ms/iter")
print(f"speedup      : {t_torch / t_flash:.2f}x")
```

Notes:

* Results vary with `B, C, N, K`, GPU model, and coefficient structure. Larger `N` and more channels tend to show bigger gains.
* The reference path uses a channel loop to honor per-channel coefficients; depending on torchaudio’s version and broadcasting rules, you may adapt the baseline to your setup.

## How it works

1. FIR stage: depthwise 1D convolution of zero-padded input with `b_norm`
2. IIR stage: causal recursion with `a_norm`
3. Backward:

   * reverse-time IIR to backprop through the recursion,
   * `convolution_backward` for FIR,
   * exact gradient handling for the `a0` normalization.

## Tips and performance notes

* `chunk_size` controls how many time steps each CUDA launch processes per block. Start with `1024–4096`.
* The kernel executes one time-sequential loop per `(B, C)` pair, which is correct but not fully GPU-saturating for tiny `B*C`.
* For CPU tensors a straightforward C++ loop is used; it favors correctness over speed.

## Troubleshooting

* `CUDA_HOME is None`: install a CUDA toolkit that matches your PyTorch build.
* `torch not found during build`: install PyTorch before this package, or use `--no-build-isolation`.
* Notebook environments: after rebuilding the extension, restart the kernel/runtime so the new `.so` is loaded.
* `a[:,0] must be non-zero`: ensure the leading denominator coefficient is not zero (and not extremely small).

## Project structure

```
flash-lfilter/
├─ flashlfilter/
│  ├─ __init__.py
│  ├─ iir_cuda_kernel.h
│  ├─ iir_cuda_kernel.cu
│  └─ lfilter_autograd.cpp
├─ setup.py
├─ README.md
├─ LICENSE
└─ .gitignore
```

## License

MIT

## Acknowledgements

Built on PyTorch’s C++/CUDA extension API (`torch.utils.cpp_extension`).
