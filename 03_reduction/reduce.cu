// Vector Reduction: compute the sum of all elements in an array.
//
// Five kernels, from slowest to fastest:
//   1. Naive      -- one thread does everything (no parallelism)
//   2. Shared     -- threads cooperate via shared memory tree reduction
//   3. Warp       -- warp shuffle instructions; one atomicAdd per warp (8/block)
//   4. Warp Block -- warp shuffle within warps, then block-level merge (one atomicAdd/block)
//   5. CUB        -- NVIDIA's production DeviceReduce (multi-element-per-thread)
//
// Why reduction matters for Flash Attention:
//   Softmax needs two reductions per row:
//     max(x)          -- for numerical stability before exp()
//     sum(exp(x-max)) -- the denominator
//   The warp shuffle pattern here is exactly what Flash Attention uses.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cub/cub.cuh>
#include "../common/helper.cuh"

#define BLOCK_SIZE 256
#define N (1 << 24)   // 16M elements (~64 MB)


// =============================================================================
// KERNEL 1 -- Naive: one thread does all the work
// =============================================================================
//
// No parallelism at all. Every other GPU thread sits completely idle.
// Only useful as a correctness baseline to compare against.
//
__global__ void reduce_naive(const float *A, float *out, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.f;
        for (int i = 0; i < n; i++) sum += A[i];
        *out = sum;
    }
}


// =============================================================================
// KERNEL 2 -- Shared memory tree reduction
// =============================================================================
//
// NEW CONCEPT: tree reduction
//
//   Instead of one thread summing everything, threads work together
//   by halving the active set each step -- like a tournament bracket.
//
//   With BLOCK_SIZE = 256 threads and 256 elements in shared memory:
//
//   step s=128: thread 0 adds sdata[128], thread 1 adds sdata[129], ...
//   step s= 64: thread 0 adds sdata[ 64], thread 1 adds sdata[ 65], ...
//   step s=  1: thread 0 adds sdata[  1]
//
//   After log2(256) = 8 steps, sdata[0] holds the sum for this block.
//   Total work: O(N) but done in O(log N) time -- massively parallel.
//
// NEW CONCEPT: atomicAdd
//
//   Many blocks finish at different times and all want to write to *out.
//   atomicAdd(ptr, val) adds val to *ptr as one uninterruptible operation --
//   no two threads can interleave their read-modify-write, so no data race.
//
__global__ void reduce_shared(const float *A, float *out, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element; pad with 0 if out of bounds.
    sdata[tid] = (i < n) ? A[i] : 0.f;
    __syncthreads();

    // Tree reduction: halve active threads each step.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 holds this block's partial sum.
    if (tid == 0)
        atomicAdd(out, sdata[0]);
}


// =============================================================================
// KERNEL 3 -- Warp shuffle reduction
// =============================================================================
//
// NEW CONCEPT: warp shuffle (__shfl_down_sync)
//
//   All 32 threads in a warp share a register file. Shuffle instructions
//   let threads read each other's registers directly -- no shared memory,
//   no __syncthreads(), just a single hardware instruction (~1 cycle).
//
//   __shfl_down_sync(mask, val, offset):
//     Thread tid receives the val of thread (tid + offset).
//     mask = 0xffffffff means all 32 lanes participate.
//
//   Five shuffles reduce 32 values to 1:
//     offset=16: lane 0  += lane 16, lane 1  += lane 17, ...
//     offset= 8: lane 0  += lane  8, lane 1  += lane  9, ...
//     offset= 4: lane 0  += lane  4, ...
//     offset= 2: lane 0  += lane  2, ...
//     offset= 1: lane 0  += lane  1
//   After 5 steps, lane 0 holds the sum of all 32 values.
//
// This exact pattern appears in Flash Attention for the online softmax step.
//
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_warp(const float *A, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? A[i] : 0.f;

    // Reduce within each warp -- lane 0 of each warp gets the warp sum.
    val = warp_reduce_sum(val);

    // Lane 0 of every warp atomically adds its partial sum.
    // threadIdx.x % 32 == 0 identifies lane 0 in each warp.
    if (threadIdx.x % 32 == 0)
        atomicAdd(out, val);
}


// =============================================================================
// KERNEL 4 -- Warp block: warp shuffle + block-level merge (one atomicAdd per block)
// =============================================================================
//
// The problem with reduce_warp: with BLOCK_SIZE=256 there are 8 warps per block,
// so 8 atomicAdds per block = 524,288 total atomics for N=16M. All serialize.
//
// Fix: after each warp reduces to its lane 0, store the 8 lane-0 values into
// shared memory and let the *first warp* reduce those with another shuffle.
// Now only thread 0 of the entire block calls atomicAdd -- same count as the
// shared-memory kernel (65,536 atomics), but with cheaper intra-block work.
//
// The first warp has 32 lanes but only 8 warp sums to reduce. Lanes 8-31
// contribute 0.f so the shuffle math stays correct across all 32 lanes.
//
__global__ void reduce_warp_block(const float *A, float *out, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];   // 8 slots for a 256-thread block

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? A[i] : 0.f;

    // Step 1: each warp reduces its 32 lanes → lane 0 holds the warp sum.
    val = warp_reduce_sum(val);

    // Step 2: lane 0 of every warp writes its partial sum to shared memory.
    if (threadIdx.x % 32 == 0)
        warp_sums[threadIdx.x / 32] = val;
    __syncthreads();

    // Step 3: first warp loads the 8 warp sums; lanes 8-31 get 0.
    //         One more shuffle → thread 0 holds the block total.
    if (threadIdx.x < 32) {
        float s = (threadIdx.x < BLOCK_SIZE / 32) ? warp_sums[threadIdx.x] : 0.f;
        s = warp_reduce_sum(s);
        if (threadIdx.x == 0)
            atomicAdd(out, s);   // one atomic per block
    }
}


// =============================================================================
// CPU reference
// =============================================================================
float reduce_cpu(const float *A, int n) {
    double sum = 0.0;   // double for accuracy with large N
    for (int i = 0; i < n; i++) sum += A[i];
    return (float)sum;
}


// =============================================================================
// main
// =============================================================================
int main() {
    const size_t nbytes = N * sizeof(float);

    float *h_A = (float *)malloc(nbytes);
    for (int i = 0; i < N; i++) h_A[i] = 1.f;   // expected sum = N = 16777216

    float ref = reduce_cpu(h_A, N);

    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float h_out, ms;
    float tol = ref * 1e-4f;

    // --- Kernel 1: Naive ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_naive<<<1, 1>>>(d_A, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Naive  : %8.3f ms   result = %.0f   %s\n",
           ms, h_out, fabsf(h_out - ref) < tol ? "PASS" : "FAIL");

    // --- Kernel 2: Shared memory ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_shared<<<grid, BLOCK_SIZE>>>(d_A, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Shared : %8.3f ms   result = %.0f   %s\n",
           ms, h_out, fabsf(h_out - ref) < tol ? "PASS" : "FAIL");

    // --- Kernel 3: Warp shuffle ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp<<<grid, BLOCK_SIZE>>>(d_A, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Warp   : %8.3f ms   result = %.0f   %s\n",
           ms, h_out, fabsf(h_out - ref) < tol ? "PASS" : "FAIL");

    // --- Kernel 4: Warp shuffle (fixed) ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp_block<<<grid, BLOCK_SIZE>>>(d_A, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("WarpBlk: %8.3f ms   result = %.0f   %s\n",
           ms, h_out, fabsf(h_out - ref) < tol ? "PASS" : "FAIL");

    // --- Kernel 5: CUB DeviceReduce ---
    //
    // cub::DeviceReduce::Sum is NVIDIA's production-quality reduction.
    // It needs a temporary workspace on the device; the first call with
    // d_temp=nullptr just queries how many bytes are required.
    //
    void  *d_temp     = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_A, d_out, N);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_A, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("CUB    : %8.3f ms   result = %.0f   %s\n",
           ms, h_out, fabsf(h_out - ref) < tol ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
