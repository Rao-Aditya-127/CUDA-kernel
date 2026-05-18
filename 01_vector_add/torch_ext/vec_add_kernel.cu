// The CUDA kernel + a C++ launcher function that PyTorch will call.
//
// Two things are different from standalone CUDA code:
//   1. We include <ATen/ATen.h> instead of <torch/extension.h> so pybind11
//      headers are not compiled by nvcc.
//   2. Instead of raw h_ / d_ pointers, we accept at::Tensor objects
//      and use .data_ptr<float>() to get the raw GPU pointer inside.

#include <ATen/ATen.h>

// ---- Kernel (unchanged from vec_add.cu) ----
__global__ void vec_add_kernel(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ---- Launcher (called from Python via pybind11) ----
//
// at::Tensor is the same type as torch::Tensor.
// .data_ptr<float>() gives the raw float* on the device -- equivalent to
// the d_A / d_B pointers we managed manually with cudaMalloc before.
//
// We return C as a Tensor so Python gets a normal torch tensor back,
// and PyTorch tracks it for autograd, memory management, etc.
//
at::Tensor vec_add_cuda(at::Tensor A, at::Tensor B) {
    const int N = A.size(0);
    auto C = at::zeros_like(A);   // allocates output on same device as A

    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vec_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
