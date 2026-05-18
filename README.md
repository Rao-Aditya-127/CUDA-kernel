# CUDA Kernels

A collection of CUDA kernels written from scratch, starting from the basics and working up to important and interesting kernels like Softmax, Flash Attention, and more.

## Structure

Each kernel lives in its own numbered folder containing the `.cu` kernel file and a `profiling.md` with profiling results and analysis.

```
CUDA_kernel/
├── 01_vector_add/          # Vector addition kernel
│   ├── vec_add.cu
│   ├── cupy_example.py
│   ├── Makefile
│   └── torch_ext/          # PyTorch extension (unique to this folder)
│       ├── vec_add_kernel.cu
│       ├── vec_add_ext.cpp
│       ├── setup.py
│       ├── run.py
│       └── README.md
├── 02_matmul/              # Matrix multiplication kernel
│   ├── matmul.cu
│   └── profiling.md
└── ...
```

## Kernels

| # | Kernel | Description |
|---|--------|-------------|
| 01 | [Vector Add](01_vector_add/) | Element-wise vector addition — the "Hello World" of CUDA |
| 02 | [Matrix Multiply](02_matmul/) | Naive and tiled matrix multiplication |
| ... | ... | More coming — Softmax, Flash Attention, and others |

## Note on `torch_ext`

The `01_vector_add` folder contains a `torch_ext/` subdirectory that demonstrates how to package a CUDA kernel as a PyTorch C++ extension so it can be called directly from Python. This is a one-time demonstration — subsequent kernel folders will not include this and will focus only on the kernel itself and its profiling results.

## Goals

- Understand GPU memory hierarchy and thread/block organization
- Profile and analyze kernel performance
- Build up to real-world kernels used in deep learning (Softmax, LayerNorm, Flash Attention, etc.)
