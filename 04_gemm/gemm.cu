/*
 * GEMM: General Matrix Multiplication
 *
 *   C = alpha * A * B + beta * C
 *
 * A is (M x K), B is (K x N), C is (M x N). Row-major storage.
 *
 * This is the full form of what we called "matmul" in 02_matmul.
 * The matmul there was a special case: alpha=1, beta=0 (fresh output buffer).
 *
 * Adding alpha and beta costs almost nothing (two FMAs per thread at the end)
 * but unlocks:
 *
 *   beta = 0  -->  C = alpha * A * B          (fresh output)
 *   beta = 1  -->  C = alpha * A * B + C      (accumulate into existing C)
 *   alpha = 1, beta = 1  -->  C += A * B      (bias addition pattern)
 *
 * This is also the signature that cuBLAS cublasSgemm() uses.
 *
 * Two kernels:
 *
 *   Kernel 1 -- Naive GEMM
 *     Same as naive matmul but with the two scalars applied at the end.
 *
 *   Kernel 2 -- Tiled GEMM
 *     Shared memory tiling (same as 02_matmul) + scalar application.
 *     This is the version worth understanding deeply -- it is the inner
 *     loop pattern that Flash Attention builds on.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../common/helper.cuh"

using namespace nvcuda;

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                    \
                    _s, __FILE__, __LINE__);                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define BLOCK_DIM 16
#define TILE_SIZE 16

// WMMA tensor-core tile shape. 16x16x16 with __half inputs + float accumulate is
// the one fragment geometry supported on BOTH Turing (T4, sm_75) and Ada (L4, sm_89).
// (TF32 16x16x8 would be cleaner for our float data but needs sm_80+, so T4 is out.)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


// =============================================================================
// KERNEL 1 -- Naive GEMM: C = alpha * A * B + beta * C
// =============================================================================
//
// Identical structure to matmul_naive from 02_matmul, except the write-back:
//
//   Before:  C[row][col] = sum
//   After:   C[row][col] = alpha * sum + beta * C[row][col]
//
// Note that we read C[row][col] BEFORE writing it when beta != 0.
// The caller must initialise C to the desired "old C" before launching.
//
__global__ void gemm_naive(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}


// =============================================================================
// KERNEL 2 -- Tiled GEMM: C = alpha * A * B + beta * C
// =============================================================================
//
// The tiling logic is identical to matmul_tiled from 02_matmul.
// The only difference is the final write:
//
//   C[row][col] = alpha * sum + beta * C[row][col]
//
// WHY DOES THIS MATTER?
//
//   In Flash Attention, attention output is computed in chunks over the
//   sequence length. Each chunk contributes a partial result that must be
//   accumulated into the output buffer with a rescaling factor.
//   That rescaling IS the beta * C term -- it lets you accumulate partial
//   GEMMs without ever materialising the full intermediate matrix.
//
__global__ void gemm_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                         ? A[row * K + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                         ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
}


// =============================================================================
// KERNEL 3 -- Tensor-Core GEMM (WMMA): C = alpha * A * B + beta * C
// =============================================================================
//
// Tensor cores are dedicated matrix-multiply-accumulate units. One instruction
// multiplies a 16x16x16 tile and accumulates it -- a whole tile per op instead
// of one scalar FMA per thread. The catch: they only consume LOW-PRECISION
// inputs. Here we use FP16 (__half) for A and B, but accumulate in FP32, which
// keeps the running sum accurate even though the operands are half precision.
//
// The unit of work is a WARP, not a thread. All 32 lanes cooperate on one
// 16x16 output tile via the opaque "fragment" types -- you never index
// individual elements of A or B by hand; load_matrix_sync / mma_sync /
// store_matrix_sync move whole tiles between memory, registers, and the cores.
//
// PRECISION NOTE: A and B are rounded to FP16 before multiplying, so the result
// differs from the exact FP32 reference by ~0.1-1%. That is expected and is why
// main() checks this kernel with a RELATIVE tolerance instead of the strict
// absolute one used for the FP32 kernels.
//
// Assumes M, N, K are multiples of 16 (true for 1024). Handling ragged edges
// would need masked loads into a padded shared-memory staging tile.
//
__global__ void gemm_wmma(
    const __half *A, const __half *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Map this warp to one 16x16 tile of C.
    //   threadIdx.x spans several warps along the row (M) direction
    //   threadIdx.y indexes the tile along the column (N) direction
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Slide a 16-wide tile across the K dimension, accumulating each product.
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;   // top row of A's tile  (in M)
        int bCol = warpN * WMMA_N;   // left col of B's tile (in N)

        if (aRow < M && bCol < N) {
            // A tile starts at (aRow, k), row stride K; B tile at (k, bCol), stride N.
            wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Apply alpha/beta and write back: C = alpha * acc + beta * C_old.
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
        for (int i = 0; i < c_frag.num_elements; i++)
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}


// Convert an FP32 buffer to FP16 on the device (one thread per element).
// Tensor cores need half-precision operands; A and B are staged through this.
__global__ void convert_f32_to_f16(const float *in, __half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}


// =============================================================================
// CPU reference
// =============================================================================
void gemm_cpu(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
}


// =============================================================================
// Helpers
// =============================================================================
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

// Relative-tolerance check for the tensor-core kernel. FP16 operands make an
// absolute 1e-2 bound unrealistic on values of magnitude ~500, so compare the
// error against the reference's own size instead.
bool verify_rel(const float *ref, const float *result, int size, float rtol = 2e-2f) {
    for (int i = 0; i < size; i++) {
        float denom = fabsf(ref[i]) > 1e-6f ? fabsf(ref[i]) : 1.0f;
        if (fabsf(ref[i] - result[i]) / denom > rtol) {
            printf("  MISMATCH at [%d]: ref=%.5f  got=%.5f  (rel=%.4f)\n",
                   i, ref[i], result[i], fabsf(ref[i] - result[i]) / denom);
            return false;
        }
    }
    return true;
}

void fill_random(float *arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX;
}


// =============================================================================
// main
// =============================================================================
int main() {
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 2.0f, beta = 0.5f;  // non-trivial to verify both scalars

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    // Host memory
    float *h_A    = (float *)malloc(bytes_A);
    float *h_B    = (float *)malloc(bytes_B);
    float *h_C0   = (float *)malloc(bytes_C);   // initial C (the "old" C for beta)
    float *h_ref  = (float *)malloc(bytes_C);
    float *h_C    = (float *)malloc(bytes_C);

    fill_random(h_A,  M * K);
    fill_random(h_B,  K * N);
    fill_random(h_C0, M * N);   // non-zero initial C to exercise beta

    // CPU reference: gemm_cpu writes into h_ref, starting from a copy of h_C0
    memcpy(h_ref, h_C0, bytes_C);
    printf("Computing CPU reference for %dx%d x %dx%d  (alpha=%.1f, beta=%.1f)...\n",
           M, K, K, N, alpha, beta);
    gemm_cpu(h_A, h_B, h_ref, M, N, K, alpha, beta);

    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM,
              (M + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Naive GEMM ---
    // Upload the initial C (so beta can act on it)
    CUDA_CHECK(cudaMemcpy(d_C, h_C0, bytes_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("\nNaive GEMM : %7.3f ms  -->  %s\n",
           elapsed_ms(start, stop),
           verify(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    // --- Tiled GEMM ---
    CUDA_CHECK(cudaMemcpy(d_C, h_C0, bytes_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("Tiled GEMM : %7.3f ms  -->  %s\n",
           elapsed_ms(start, stop),
           verify(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    // --- Tensor-Core GEMM (WMMA, FP16 inputs) ---
    //
    // Tensor cores need FP16 operands, so stage half-precision copies of A and B.
    __half *d_A_h, *d_B_h;
    CUDA_CHECK(cudaMalloc(&d_A_h, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_h, K * N * sizeof(__half)));

    int convBlock = 256;
    convert_f32_to_f16<<<(M * K + convBlock - 1) / convBlock, convBlock>>>(d_A, d_A_h, M * K);
    convert_f32_to_f16<<<(K * N + convBlock - 1) / convBlock, convBlock>>>(d_B, d_B_h, K * N);
    CUDA_CHECK(cudaGetLastError());

    // One warp computes one 16x16 tile. Block = 128x4 threads = 4x4 = 16 warps,
    // each warp owning a 16x16 tile, so a block covers a 64x64 region of C.
    dim3 wmmaBlock(128, 4);
    dim3 wmmaGrid((M + (WMMA_M * wmmaBlock.x / 32) - 1) / (WMMA_M * wmmaBlock.x / 32),
                  (N + (WMMA_N * wmmaBlock.y) - 1) / (WMMA_N * wmmaBlock.y));

    CUDA_CHECK(cudaMemcpy(d_C, h_C0, bytes_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    gemm_wmma<<<wmmaGrid, wmmaBlock>>>(d_A_h, d_B_h, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("WMMA (FP16): %7.3f ms  -->  %s  (relative tol)\n",
           elapsed_ms(start, stop),
           verify_rel(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    // --- cuBLAS GEMM ---
    //
    // cublasSgemm expects column-major matrices, but ours are row-major.
    // Trick: exploit the identity  C = alpha*A*B + beta*C
    //                          =>  C^T = alpha*B^T*A^T + beta*C^T
    // Passing B where cuBLAS expects its "A" and A where it expects its "B"
    // (with no-transpose ops on both) makes cuBLAS compute exactly what we want.
    //
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // Computes C = alpha * op(A)[m×k] * op(B)[k×n] + beta * C[m×n]  (col-major)
    //
    // Our mapping:
    //   "A" = d_B (row-major B treated as col-major B^T),  lda = N  (row width of B)
    //   "B" = d_A (row-major A treated as col-major A^T),  ldb = K  (row width of A)
    //   m=N, n=M, k=K  (dimensions of the col-major result C^T which is M×N row-major)
    //
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Warmup: first cuBLAS call triggers JIT kernel selection and compilation.
    // Without this, the timer captures library init, not the GEMM itself.
    CUBLAS_CHECK(cublasSgemm(cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_C, h_C0, bytes_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("cuBLAS     : %7.3f ms  -->  %s\n",
           elapsed_ms(start, stop),
           verify(h_ref, h_C, M * N) ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(cublas));

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A); free(h_B); free(h_C0); free(h_ref); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_h));
    CUDA_CHECK(cudaFree(d_B_h));

    return 0;
}
