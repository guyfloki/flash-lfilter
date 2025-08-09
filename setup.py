from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

PACKAGE = "flashlfilter"

if CUDA_HOME is None:
    raise RuntimeError(
        "CUDA toolkit not found (CUDA_HOME is None). "
        "Install a CUDA toolkit compatible with your PyTorch build, "
        "or install this package in an environment with CUDA available. "
        "This setup.py does not target CPU-only builds."
    )

readme = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    readme = readme_path.read_text(encoding="utf-8")

setup(
    name="flash-lfilter",
    version="1.0.0",
    description="Fast IIR filter (lfilter) with PyTorch CUDA and autograd",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Liubomyr Horbatko",
    url="https://github.com/guyfloki/flash-lfilter",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    package_data={PACKAGE: ["*.h"]},
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE}.ops",
            sources=[
                f"{PACKAGE}/iir_cuda_kernel.cu",
                f"{PACKAGE}/lfilter_autograd.cpp",
            ],
            include_dirs=[PACKAGE],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-std=c++17", "-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
