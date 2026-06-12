# Module 04 — GEMM

## What is GEMM?

GEMM stands for **General Matrix Multiplication**. The full form is:

```
C = alpha * A * B + beta * C
```

where `alpha` and `beta` are scalar multipliers. `A` is `(M × K)`,
`B` is `(K × N)`, and `C` is `(M × N)`.

In module 02 (matmul) we computed the simpler `C = A * B`, which is just GEMM
with `alpha = 1` and `beta = 0`. Adding the two scalars costs almost nothing —
two extra multiplications per thread at the very end — but unlocks several
important usage patterns:

```
alpha=1, beta=0  →  C  = A * B          (fresh output, same as matmul)
alpha=1, beta=1  →  C += A * B          (accumulate into an existing buffer)
alpha=2, beta=0  →  C  = 2 * A * B      (scaled output)
alpha=1, beta=0.5 → C  = A * B + 0.5*C  (blend new result with old)
```

This is also the exact signature of `cublasSgemm()` — NVIDIA's production
BLAS routine. Understanding GEMM means understanding the interface that every
deep learning framework calls under the hood.

---

## How the Computation Works

Each element of `C` is a **dot product** of one row of `A` with one column of `B`,
scaled by `alpha`, then added to the scaled old value of `C`.

```
C[row][col] = alpha * dot(A[row, :], B[:, col]) + beta * C[row][col]
```

For a 4×4 example with `alpha=1, beta=0`:

```
A (4×3):          B (3×4):            C[0][0] = A[0]·B[:,0]
[ 1  2  3 ]    [ 7  8  9 10 ]              = 1×7 + 2×11 + 3×15
[ 4  5  6 ]    [11 12 13 14 ]              = 7 + 22 + 45
[ 7  8  9 ]    [15 16 17 18 ]              = 74
[10 11 12 ]
```

Every element of the `M × N` output matrix C requires `K` multiply-accumulate
operations. Total work: `M × N × K` multiply-adds, which for a 1024³ matrix
is ~2.1 billion operations.

---

## Kernel 1 — Naive GEMM

```
Thread assignment: one thread computes one element of C.

Thread (row, col):
  sum = 0
  for k in 0..K:
      sum += A[row][k] * B[k][col]
  C[row][col] = alpha * sum + beta * C[row][col]
```

The structure is identical to naive matmul. The only addition is the final line,
which reads the old `C[row][col]` before overwriting it (needed when `beta != 0`).

**The memory access problem:**

For matrix `B`, thread `(row, col)` accesses `B[k][col]` — stepping down column
`col` across rows. Threads in the same warp have different `col` values, so they
access different columns of `B` simultaneously. These columns are stored far apart
in row-major memory, producing strided accesses that waste cache lines.

---

## Kernel 2 — Tiled GEMM

The tiled kernel fixes the memory problem by loading chunks of `A` and `B` into
shared memory before computing, so every global memory load is reused many times.

**The idea — divide C into 16×16 tiles:**

Each block is responsible for computing one 16×16 tile of the output `C`.
To compute that tile, it needs to multiply a row-strip of `A` with a column-strip
of `B`. These strips are consumed in 16-column chunks called **tiles**.

```
A (M×K):                 B (K×N):
┌──────────┬────────┐    ┌────┬────┬────┐
│ tile_A   │        │    │    │    │    │
│ (16×16)  │  ...   │ ×  │t_B │ .. │    │ = one 16×16 tile of C
└──────────┴────────┘    └────┴────┴────┘
   step t=0                step t=0
```

**Each iteration of the tile loop:**

```
Step 1: all 256 threads cooperatively load a 16×16 tile of A into shared memory
        and a 16×16 tile of B into shared memory
        __syncthreads() -- wait until all loads are done

Step 2: each thread computes its dot product contribution using the tiles
        (16 multiply-adds from shared memory instead of global memory)

Step 3: __syncthreads() -- wait before overwriting tiles in the next iteration

Repeat for all K/16 tile steps, accumulating into sum.
```

After all tile steps: `C[row][col] = alpha * sum + beta * C[row][col]`

**Why this is faster:**

In the naive kernel, each element of `B` is read once per thread that needs it.
With a 16×16 tile, each value loaded into `tile_B` is reused by 16 threads (one
per row of the output tile). Each global memory load is amortised across 16 operations
instead of just 1 — 16× fewer global reads for `B`.

```
Global memory reads per output element:
  Naive :  2 × K       = 2048  (every thread reads its own copy)
  Tiled :  2 × K / 16  =  128  (tile is shared across 16 threads)
```

---

## The alpha / beta Write-back

The final line of both kernels:

```cu
C[row * N + col] = alpha * sum + beta * C[row * N + col];
```

When `beta != 0` this reads the old value of `C` before writing. This means:

1. The caller must initialise `C` on the device before launching the kernel.
2. The read and write to `C[row][col]` are by the same thread — no race condition.

Setting `beta = 0` makes the read a multiply-by-zero that the compiler optimises
away, so there is no performance penalty for the common case.

---

## Why This Matters for Flash Attention

Standard attention requires computing:

```
scores = Q * K^T          (GEMM)
weights = softmax(scores)
output = weights * V      (GEMM)
```

The problem: `scores` is a sequence_length × sequence_length matrix. For long
sequences this does not fit in GPU memory and is extremely slow to write and
read back.

Flash Attention avoids materialising `scores` by processing `Q`, `K`, and `V`
in blocks and **accumulating partial results directly into the output buffer**
using a rescaling factor. That rescaling IS the `beta * C` term:

```
output_block += rescale_factor * partial_attention_result
```

Every iteration updates the output with a new partial result weighted by a
factor that corrects for the softmax normalisation computed so far. This is
exactly `C = alpha * partial + beta * C_so_far` — repeated for each block.

Understanding tiled GEMM with alpha/beta is therefore the direct prerequisite
for understanding how Flash Attention avoids the quadratic memory cost of full
attention.

---

## Comparison

| | Naive | Tiled |
|---|---|---|
| Global reads per output element | 2K | 2K / TILE_SIZE |
| Shared memory used | none | 2 × 16×16 × 4 = 2 KB/block |
| Sync barriers per tile step | none | 2 × `__syncthreads()` |
| Cache line efficiency (B matrix) | poor (column access) | good (tile is loaded once) |

---

## Kernel 3 — Tensor-Core GEMM (WMMA)

Tensor cores are dedicated matrix-multiply-accumulate units that multiply a whole
`16×16×16` tile per instruction instead of doing one scalar FMA per thread. The
trade-off is precision: they only consume **low-precision inputs**. This kernel
feeds them **FP16** copies of `A` and `B` while accumulating in **FP32**, so the
running sum stays accurate even though the operands are half precision.

```
Unit of work:   a WARP (32 lanes), not a thread.
Data type:      __half inputs, float accumulator.
API:            wmma::load_matrix_sync / mma_sync / store_matrix_sync
                operate on opaque "fragment" tiles — you never index A or B
                element-by-element.
```

**Why FP16 (and not TF32)?** The target VM GPUs are the T4 (Turing, `sm_75`) and
the L4 (Ada, `sm_89`). TF32 tensor cores only exist on `sm_80+`, so they would
exclude the T4. The `16×16×16` FP16 fragment shape is the one geometry supported
on **both**, so it is the portable choice.

**Precision check:** because `A` and `B` are rounded to FP16, the result differs
from the exact FP32 reference by ~0.1–1%. The strict absolute `1e-2` check used
for the FP32 kernels would wrongly FAIL it, so this kernel is verified with a
**relative tolerance** (`verify_rel`, 2%) instead.

This kernel assumes `M`, `N`, `K` are multiples of 16 (true for 1024). Ragged
edges would need masked loads into a padded staging tile.

---

---

## Kernel 4 — Shared-Memory Tensor-Core GEMM (WMMA + staging)

Profiling Kernel 3 on a T4 showed **both** pipes nearly idle — ~6% tensor-core
(HMMA) activity and ~10% DRAM throughput. Neither compute- nor bandwidth-bound:
the kernel is **latency-bound**. Each warp loads an A/B tile straight from global
memory and immediately feeds a *dependent* `mma_sync`, so the tensor cores spend
almost all their time stalled waiting on those loads.

Kernel 4 applies the three levers a tuned GEMM uses to keep the cores fed:

```
1. REUSE         The whole block cooperatively stages a 64×16 slab of A and a
                 16×64 slab of B into shared memory once per K-step. Fragment
                 loads then hit fast shared memory, and each staged value is
                 reused by many warps instead of being re-fetched from DRAM.

2. ILP           Each warp owns a 32×32 output region = a 2×2 grid of accumulator
                 fragments → 4 INDEPENDENT mma_sync ops per K-step. While one MMA
                 waits on operands, another can issue. Kernel 3's single
                 accumulator had nothing to overlap.

3. LESS TRAFFIC  An A operand staged in shared memory feeds both column-tiles a
                 warp owns (and a B operand feeds both row-tiles), cutting global
                 reads further.
```

```
Block: 128 threads = 4 warps, arranged 2×2.
Tile : block computes a 64×64 tile of C; warp (wr,wc) owns its 32×32 quadrant.
Smem : 2 KB (A slab) + 2 KB (B slab) = 4 KB per block.
```

The expected payoff is the **HMMA-pipe utilization climbing** from ~6% toward the
tens-of-percent range as the stalls disappear — re-run the `ncu` command from the
profiling notes on `gemm_wmma_smem` to watch it move. The remaining gap to cuBLAS
is mostly double-buffering (prefetching the next K-slab while computing the
current one) and larger warp tiles, which this kernel keeps simple on purpose.

---

## Files in This Module

```
04_gemm/
    gemm.cu    Five kernels with timing and correctness check:
               naive, tiled, WMMA (tensor core), WMMA+smem, cuBLAS
```

## Build & Run

```bash
cd 04_gemm
# L4 (Ada):    -arch=sm_89
# T4 (Turing): -arch=sm_75
nvcc -O2 -arch=sm_89 -I.. gemm.cu -o gemm -lcublas
./gemm
```

The WMMA path needs the `<mma.h>` / `<cuda_fp16.h>` headers (bundled with CUDA)
and a real `-arch` that has tensor cores — `sm_75` or higher. Compiling for a
pre-Turing arch will fail to find the WMMA intrinsics.

## What to Notice

- Tiled is faster than naive — same reason as in 02_matmul: shared memory
  eliminates redundant global reads of `B`
- cuBLAS will be significantly faster than both — it uses highly tuned register
  tiling, vectorised loads, and tensor cores on supported hardware
- All three produce identical results — the CPU reference uses `alpha=2, beta=0.5`
  with a non-zero initial `C` to verify that both scalars are applied correctly
- cuBLAS expects column-major matrices; we pass B where it expects A and vice versa,
  exploiting `(AB)^T = B^T A^T` to get the correct row-major result without any
  data transposition
- The tiled kernel is the inner loop pattern that Flash Attention builds on —
  the `beta * C` accumulation is not cosmetic, it is the mechanism that allows
  multi-pass attention without storing the full score matrix
