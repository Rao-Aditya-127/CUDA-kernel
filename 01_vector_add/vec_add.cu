/*
 * Vector Addition: C[i] = A[i] + B[i]
 *
 * This is the "Hello World" of CUDA. It demonstrates the core workflow:
 *   1. Allocate memory on both CPU (host) and GPU (device)
 *   2. Copy input data from host -> device
 *   3. Launch a kernel -- the GPU runs it in parallel across thousands of threads
 *   4. Copy results back from device -> host
 *   5. Verify and free memory
 *
 * Naming convention used throughout:
 *   h_* = host (CPU) pointer    e.g. h_A
 *   d_* = device (GPU) pointer  e.g. d_A
 */

#include <cstdio>
#include <cstdlib>
#include "../common/helper.cuh"   // CUDA_CHECK macro for error handling


// =============================================================================
// KERNEL  (runs on GPU)
// =============================================================================
//
// __global__ marks this as a CUDA kernel: called from CPU, executes on GPU.
//
// The GPU launches this function for every thread simultaneously.
// With 1M elements and 256 threads/block we get ~4096 blocks, each running
// 256 threads -- so ~1M threads run this function at the same time.
//
// Each thread figures out which element it owns using three built-in variables:
//
//   blockIdx.x  = index of this block  within the grid  (0 ... GRID_SIZE-1)
//   blockDim.x  = number of threads    per block        (= BLOCK_SIZE = 256)
//   threadIdx.x = index of this thread within its block (0 ... 255)
//
// Putting them together:
//
//   block 0: threads 0..255  ->  global indices   0..255
//   block 1: threads 0..255  ->  global indices 256..511
//   block k: threads 0..255  ->  global indices k*256 .. k*256+255
//
//   i = blockIdx.x * blockDim.x + threadIdx.x   <-- unique index per thread
//
// The bounds check (i < N) handles the case where N is not a perfect multiple
// of BLOCK_SIZE -- the extra threads in the last block just do nothing.
//
__global__ void vec_add(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index

    if (i < N) {          // guard: don't go out of bounds
        C[i] = A[i] + B[i];
    }
}


// =============================================================================
// HOST (runs on CPU)
// =============================================================================

int main() {

    // --- Problem size ---------------------------------------------------------
    const int    N      = 1 << 20;          // 2^20 = 1,048,576 elements (~1M)
    const size_t nbytes = N * sizeof(float);

    // --- Launch configuration -------------------------------------------------
    //
    // CUDA organises threads into a two-level hierarchy:
    //
    //   Grid  = collection of blocks
    //   Block = collection of threads (up to 1024 per block)
    //
    // A block runs on a single SM (Streaming Multiprocessor). Threads within
    // a block can share memory and synchronise; threads across blocks cannot.
    //
    // 256 threads/block is a common default -- it is a multiple of the warp
    // size (32) and leaves room for the hardware to hide memory latency by
    // switching between warps.
    //
    // We need enough blocks so that blockCount * threadsPerBlock >= N.
    // The ceiling division (N + BLOCK_SIZE - 1) / BLOCK_SIZE avoids needing
    // a separate remainder block.
    //
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // ceil(N / 256)


    // --- Allocate host (CPU) memory -------------------------------------------
    float *h_A = (float *)malloc(nbytes);
    float *h_B = (float *)malloc(nbytes);
    float *h_C = (float *)malloc(nbytes);   // output; filled after kernel runs

    // --- Initialise inputs ----------------------------------------------------
    // Fill A with 1.0 and B with 2.0 so the expected result is 3.0 everywhere.
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    // --- Allocate device (GPU) memory -----------------------------------------
    //
    // cudaMalloc works like malloc but allocates on the GPU.
    // The CPU and GPU have separate address spaces -- a CPU pointer and a GPU
    // pointer cannot be dereferenced on the wrong side.
    //
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CHECK(cudaMalloc(&d_B, nbytes));
    CUDA_CHECK(cudaMalloc(&d_C, nbytes));


    // --- Copy inputs: host -> device ------------------------------------------
    //
    // cudaMemcpy(dst, src, bytes, direction)
    // The kernel can only read from GPU memory, so we must transfer first.
    //
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice));


    // --- Launch kernel --------------------------------------------------------
    //
    // Syntax:  kernel<<<gridSize, blockSize>>>(args...)
    //
    // This is asynchronous -- the CPU does NOT wait here; it continues to the
    // next line immediately while the GPU executes the kernel in parallel.
    //
    vec_add<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for errors in the kernel launch itself (bad config, OOM, etc.)
    CUDA_CHECK(cudaGetLastError());

    // Block the CPU here until the GPU has finished all pending work.
    // Without this, the cudaMemcpy below could run before the kernel finishes.
    CUDA_CHECK(cudaDeviceSynchronize());


    // --- Copy result: device -> host ------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost));


    // --- Verify ---------------------------------------------------------------
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");


    // --- Free memory ----------------------------------------------------------
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
