/*
 * Matrix Multiplication: C = A * B
 *
 * A is (M x K), B is (K x N), C is (M x N).
 * All matrices stored in row-major order in memory.
 *
 * Two implementations:
 *
 *   Kernel 1 -- Naive
 *     Each thread computes one output element by reading directly from
 *     global memory. Simple, but every read goes to slow DRAM.
 *
 *   Kernel 2 -- Tiled (shared memory)
 *     Threads in a block cooperate to load chunks ("tiles") of A and B
 *     into fast on-chip shared memory, then compute from there.
 *     This is the core idea behind Flash Attention.
 *
 * Run both and observe the timing difference.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../common/helper.cuh"

#define BLOCK_DIM 16   // threads per block dimension (both kernels)
#define TILE_SIZE 16   // shared memory tile size (tiled kernel only; equals BLOCK_DIM intentionally)


// =============================================================================
// KERNEL 1 -- Naive matrix multiplication
// =============================================================================
//
// NEW CONCEPT: 2D thread layout
//
//   In vec_add we used a 1D grid because each element had one index (i).
//   A matrix element has two indices (row, col), so we use a 2D grid.
//
//   dim3 block(TILE_SIZE, TILE_SIZE)  -->  a block is a 2D tile of threads
//   dim3 grid(ceil(N/T), ceil(M/T))   -->  grid covers the whole output matrix
//
//   Inside the kernel:
//     col = blockIdx.x * blockDim.x + threadIdx.x   (x moves along columns)
//     row = blockIdx.y * blockDim.y + threadIdx.y   (y moves along rows)
//
//   Each thread then owns exactly one output element C[row][col].
//
// WHY IS THIS SLOW?
//   To compute C[row][col], a thread reads K values from row `row` of A
//   and K values from column `col` of B -- all from global memory (DRAM).
//   For a 1024x1024 matrix that is over 2 billion global memory reads.
//   Global memory latency is ~400-800 cycles; shared memory is ~4 cycles.
//
__global__ void matmul_naive(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;   // out-of-bounds guard

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // Row-major indexing:
        //   A[row][k]   = A[row * K + k]
        //   B[k][col]   = B[k  * N + col]
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}


// =============================================================================
// KERNEL 2 -- Tiled matrix multiplication (shared memory)
// =============================================================================
//
// NEW CONCEPT: shared memory  (__shared__)
//
//   __shared__ allocates memory on the SM's on-chip SRAM, not in DRAM.
//   It is shared by all threads within the same block (not across blocks).
//   Bandwidth is ~100x higher than global memory.
//
// HOW TILING WORKS:
//
//   We split the K dimension into tiles of width TILE_SIZE.
//   For each tile t:
//     (a) All TILE_SIZE*TILE_SIZE threads cooperate to load one tile of A
//         and one tile of B into shared memory. Each thread loads 1 element.
//     (b) All threads compute their partial dot product using those tiles.
//     (c) Move to the next tile.
//
//   Memory savings:
//     Naive:  each thread reads 2*K global elements independently.
//     Tiled:  the block loads 2*K*TILE_SIZE global elements once, shared
//             across TILE_SIZE^2 threads.
//             Per-thread cost = 2*K/TILE_SIZE  -->  16x fewer global reads.
//
// NEW CONCEPT: __syncthreads()
//
//   A block-wide barrier. No thread proceeds past this line until every
//   thread in the block has reached it.
//
//   We need it TWICE per tile iteration:
//
//   (1) After loading  -- ensures the tile is fully written before anyone reads.
//       Without this: thread 0 might start computing while thread 15 is still
//       writing its element into shared memory.
//
//   (2) After computing -- ensures all reads from the tile are done before
//       any thread overwrites it in the next iteration.
//       Without this: fast threads could start loading tile t+1, corrupting
//       data that slow threads are still reading from tile t.
//
__global__ void matmul_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    // On-chip shared memory tiles.
    // Both are fixed at TILE_SIZE x TILE_SIZE floats.
    // They persist for the lifetime of the block (across loop iterations).
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Global row and column this thread is responsible for in C.
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Walk through the K dimension tile by tile.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // ── Load phase ────────────────────────────────────────────────────────
        // Each thread loads ONE element into tile_A and ONE into tile_B.
        //
        // tile_A holds a (TILE_SIZE x TILE_SIZE) slice of A:
        //   rows:    [blockIdx.y*TILE_SIZE .. blockIdx.y*TILE_SIZE + TILE_SIZE)
        //   columns: [t*TILE_SIZE          .. t*TILE_SIZE + TILE_SIZE)
        //
        // tile_B holds a (TILE_SIZE x TILE_SIZE) slice of B:
        //   rows:    [t*TILE_SIZE          .. t*TILE_SIZE + TILE_SIZE)
        //   columns: [blockIdx.x*TILE_SIZE .. blockIdx.x*TILE_SIZE + TILE_SIZE)
        //
        int a_col = t * TILE_SIZE + threadIdx.x;  // column in A for this thread
        int b_row = t * TILE_SIZE + threadIdx.y;  // row    in B for this thread

        // Boundary check: if K is not a multiple of TILE_SIZE, the last tile
        // may extend out of bounds -- pad with 0 so it doesn't affect the sum.
        tile_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                         ? A[row * K + a_col] : 0.0f;

        tile_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                         ? B[b_row * N + col] : 0.0f;

        // (1) Wait for all threads to finish loading before anyone computes.
        __syncthreads();

        // ── Compute phase ─────────────────────────────────────────────────────
        // Accumulate the dot product for this tile.
        // All reads are from shared memory -- fast!
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // (2) Wait for all threads to finish reading before the next iteration
        //     overwrites the shared tiles.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// =============================================================================
// CPU reference (used to verify GPU results)
// =============================================================================
void matmul_cpu(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
}


// =============================================================================
// Helpers
// =============================================================================

// Elapsed time between two CUDA events in milliseconds.
// NEW CONCEPT: CUDA events
//   cudaEventRecord() stamps a point in the GPU command stream.
//   cudaEventElapsedTime() measures the GPU time between two stamps.
//   This is the correct way to time GPU kernels -- using CPU timers like
//   clock() would also include cudaDeviceSynchronize() overhead.
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

bool verify(const float *ref, const float *result, int size, float tol = 1e-2f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(ref[i] - result[i]) > tol) {
            printf("  MISMATCH at [%d]: ref=%.5f  got=%.5f\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}


// =============================================================================
// main
// =============================================================================
int main() {
    const int M = 1024, N = 1024, K = 1024;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    // --- Host memory ---
    float *h_A   = (float *)malloc(bytes_A);
    float *h_B   = (float *)malloc(bytes_B);
    float *h_ref = (float *)malloc(bytes_C);
    float *h_C   = (float *)malloc(bytes_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    printf("Computing CPU reference for %dx%d x %dx%d ...\n", M, K, K, N);
    matmul_cpu(h_A, h_B, h_ref, M, N, K);

    // --- Device memory ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // --- 2D launch config ---
    // block: BLOCK_DIM x BLOCK_DIM threads (256 total)
    // grid:  enough blocks to cover the full M x N output matrix
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM,    // x = column blocks
              (M + BLOCK_DIM - 1) / BLOCK_DIM);   // y = row blocks

    // --- CUDA events for timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Run naive ---
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("\nNaive : %7.3f ms  -->  %s\n",
           elapsed_ms(start, stop),
           verify(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    // --- Run tiled ---
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("Tiled : %7.3f ms  -->  %s\n",
           elapsed_ms(start, stop),
           verify(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A); free(h_B); free(h_ref); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
