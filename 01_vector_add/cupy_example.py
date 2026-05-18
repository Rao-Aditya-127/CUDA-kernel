"""
Calling a custom CUDA kernel from Python using CuPy.

CuPy compiles the kernel string at runtime using nvcc under the hood.
It caches the compiled result, so subsequent runs are fast.

Install: pip install cupy-cuda12x  (match your CUDA version)
"""

import cupy as cp

# The kernel is written as a plain Python string -- identical to what you'd
# write in a .cu file. CuPy compiles it on first use.
KERNEL_CODE = """
__global__ void vec_add(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
"""

# RawKernel(source_code, kernel_function_name)
vec_add = cp.RawKernel(KERNEL_CODE, "vec_add")

# CuPy arrays live on the GPU automatically -- no manual cudaMalloc needed.
N = 1 << 20   # 1M elements
A = cp.ones(N, dtype=cp.float32)
B = cp.full(N, 2.0, dtype=cp.float32)
C = cp.zeros(N, dtype=cp.float32)

# Launch: vec_add((grid,), (block,), (args...))
BLOCK_SIZE = 256
GRID_SIZE  = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
vec_add((GRID_SIZE,), (BLOCK_SIZE,), (A, B, C, N))

# cp.asnumpy() copies back to CPU for verification
result = cp.asnumpy(C)
print("Max error:", abs(result - 3.0).max())   # should print 0.0
