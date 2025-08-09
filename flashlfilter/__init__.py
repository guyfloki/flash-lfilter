from importlib import import_module
import warnings as _warnings
import torch as _torch

try:
    import_module("flashlfilter.ops")  
except Exception as e:
    _warnings.warn(
        "flashlfilter: native extension wasn't imported. "
        "Ensure PyTorch is installed and build toolchain is available. "
        f"Original error: {e}"
    )

def lfilter_autograd(waveform, a, b, chunk_size: int):
    return _torch.ops.flashlfilter.lfilter_autograd(waveform, a, b, chunk_size)

__all__ = ["lfilter_autograd"]
