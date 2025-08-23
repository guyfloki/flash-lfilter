# flash-lfilter

Fast per-channel IIR filtering for PyTorch with CUDA and full autograd support.
Works like `lfilter`: a depthwise FIR stage followed by a causal IIR recursion.

## Installation
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
