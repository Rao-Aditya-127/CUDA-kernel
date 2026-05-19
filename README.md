# CUDA Kernels

A collection of CUDA kernels written from scratch, starting from the basics and working up to important kernels used in deep learning — Softmax, Flash Attention, and more.

## Structure

Each kernel lives in its own numbered folder containing the `.cu` kernel file, a `README.md` explaining the concepts, and a `profiling.md` with profiling results and analysis.

```
CUDA_kernel/
├── 01_vector_add/
│   ├── vec_add.cu
│   ├── cupy_example.py
│   ├── Makefile
│   └── torch_ext/              # PyTorch extension (unique to this folder)
│       ├── vec_add_kernel.cu
│       ├── vec_add_ext.cpp
│       ├── setup.py
│       ├── run.py
│       └── README.md
├── 02_matmul/
│   ├── matmul.cu
│   └── profiling.md
├── 03_reduction/
│   ├── reduce.cu
│   ├── README.md
│   └── profiling.md
└── 04_gemm/
    ├── gemm.cu
    ├── README.md
    └── profiling.md
```

## Kernels

| # | Kernel | Concepts | vs Library |
|---|--------|----------|------------|
| 01 | [Vector Add](01_vector_add/) | Thread indexing, basic memory transfers | — |
| 02 | [Matrix Multiply](02_matmul/) | Shared memory tiling, cache reuse | — |
| 03 | [Reduction](03_reduction/) | Tree reduction, warp shuffles, atomic contention | vs CUB |
| 04 | [GEMM](04_gemm/) | alpha/beta scalars, tiled GEMM, register tiling gap | vs cuBLAS |

## Key findings so far

**Reduction (03):** The naive warp shuffle kernel is *slower* than the shared memory
tree reduction despite doing less work — 524,288 atomicAdds all targeting one address
serialize at the L2 cache (90.17% long scoreboard stall). Fixing this with one extra
shared memory step (`reduce_warp_block`) cuts atomics 8× and recovers 2.56×.

**GEMM (04):** The tiled kernel's dominant bottleneck is shared memory, not global
memory — MIO throttle hits 49.73% because a 16×16 tile gives only 1 FMA per shared
memory read. cuBLAS uses 128×64 tiles, reads 4× less DRAM, executes 3× fewer
instructions, and achieves this at only 46.74% occupancy — demonstrating that
occupancy is not performance.

## Note on `torch_ext`

The `01_vector_add` folder contains a `torch_ext/` subdirectory demonstrating how
to package a CUDA kernel as a PyTorch C++ extension. This is a one-time demonstration
— subsequent folders focus on the kernel itself and its profiling results.

## Goals

- Understand GPU memory hierarchy and thread/block organisation
- Profile kernels with Nsight Compute and interpret the bottlenecks
- Build up to real-world kernels used in deep learning (Softmax, LayerNorm, Flash Attention)
