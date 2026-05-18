"""
Build the extension with:

    python setup.py install

Or for a faster development loop (no install, just compiles):

    python setup.py build_ext --inplace

After building, 'import vec_add_ext' works from this directory.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vec_add_ext",
    ext_modules=[
        CUDAExtension(
            name="vec_add_ext",           # the Python import name
            sources=[
                "vec_add_ext.cpp",        # pybind11 bindings
                "vec_add_kernel.cu",      # CUDA kernel + launcher
            ],
            extra_compile_args={
                "cxx":  ["-O2"],
                "nvcc": ["-O2", "-arch=sm_89"],  # RTX 2000 Ada, compute cap 8.9
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
