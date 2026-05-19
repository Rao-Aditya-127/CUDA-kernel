# GEMM — Kernel Profiling

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA T4 (CC 7.5) |
| Matrix size | 1024 × 1024 × 1024 (M=K=N=1024) |
| alpha / beta | 2.0 / 0.5 (non-trivial to exercise both scalars) |
| Block size | 16 × 16 = 256 threads |
| Tile size | 16 × 16 |
| Compile flags | `-O2 -arch=sm_75 -lineinfo` |

---

## Results Summary

```
Naive GEMM :   3.534 ms   PASS
Tiled GEMM :   2.182 ms   PASS
cuBLAS     :   0.340 ms   PASS

Tiled  vs Naive  :  1.62× faster
cuBLAS vs Tiled  :  6.42× faster
cuBLAS vs Naive  : 10.39× faster
```

| Metric | Naive | Tiled 16×16 | cuBLAS 128×64 |
|---|---|---|---|
| **Duration — ncu (ms)** | 3.51 | 2.15 | 0.33 |
| **DRAM throughput %** | 19.15 | 24.95 | 59.47 |
| **SM throughput %** | 60.78 | 73.60 | 83.19 |
| **DRAM bytes read (MB)** | 206 | 165 | 40 |
| **Achieved occupancy %** | 98.67 | 98.66 | 46.74 |
| **Long scoreboard stall %** | 7.22 | 18.81 | 9.93 |
| **MIO throttle stall %** | 0.05 | **49.73** | 9.12 |
| **Wait stall % (`__syncthreads`)** | 5.42 | 6.56 | 12.51 |
| **Total instructions** | 156.7M | 129.1M | 42.4M |

---

## 1. Execution Time

```
Naive  :  3.534 ms   (each thread computes one output element, K=1024 serial multiply-adds)
Tiled  :  2.182 ms   (1.62× faster — fewer instructions, less DRAM traffic)
cuBLAS :  0.340 ms   (6.42× faster than Tiled, 10.4× faster than Naive)
```

The 1.62× gain from Naive→Tiled is smaller than you might expect. The tiled kernel
reduces DRAM traffic and instruction count, but introduces a new dominant bottleneck —
the shared memory pipeline — that partially cancels the gain. The full story requires
the stall data to understand.

The 6.42× gain from Tiled→cuBLAS comes from a fundamentally different tile strategy:
cuBLAS uses 128×64 tiles where our kernel uses 16×16, giving 8× more reuse per global
load and 3× fewer total instructions.

---

## 2. Memory Bandwidth

```
                       Naive    Tiled 16×16    cuBLAS 128×64
DRAM throughput %  :   19.15      24.95           59.47
DRAM bytes read    :   206 MB     165 MB           40 MB
Theoretical min    :    ~8 MB      ~8 MB            ~8 MB
```

All three kernels perform the same mathematical work on the same data (A: 4 MB,
B: 4 MB, C: 4 MB — ~12 MB total). The difference in DRAM bytes read shows how
efficiently each kernel reuses data from cache and shared memory.

**Naive — 206 MB for 8 MB of unique data (26× over-read):**

Each of the M×N = 1M threads reads an independent column of B across K=1024 rows.
Different threads in the same warp read the same B column — and the L1/L2 cache
absorbs some of this reuse — but the cache is not large enough to hold all of B
(4 MB) while also serving A. Cache lines are evicted and re-fetched repeatedly,
causing 26× more DRAM traffic than the theoretical minimum.

**Tiled 16×16 — 165 MB (21× over-read):**

Each 16×16 tile of B is loaded once from global memory and reused 16 times across
the 16 threads in that tile column. This gives a 16× reuse factor — but only for
the data within the shared memory tile. Tiles outside the current window are still
fetched from DRAM. The improvement is real (206→165 MB, 20% less) but modest
because 16×16 tiles are small relative to the total data.

**cuBLAS 128×64 — 40 MB (5× over-read):**

With 128-wide tiles, each element of B is reused 128 times instead of 16. cuBLAS
reaches 5× the theoretical minimum — still not perfect (due to boundary alignment
and partial tiles), but 4× more efficient than our tiled kernel. This is why its
DRAM throughput (59.47%) is 2.4× higher than ours (24.95%) despite reading less
data: it actually saturates the memory pipeline when it does read.

---

## 3. Warp Stall Analysis

```
                           Naive    Tiled 16×16    cuBLAS 128×64
Long scoreboard %       :   7.22       18.81           9.93
MIO throttle %          :   0.05       49.73           9.12
Wait % (__syncthreads)  :   5.42        6.56          12.51
Total instructions      : 156.7M      129.1M          42.4M
```

**What each stall type means:**

| Stall | Cause |
|---|---|
| Long scoreboard | Thread issued a global memory load and is waiting for it to return from L2/DRAM |
| MIO throttle | The shared memory pipeline is saturated — thread has an instruction ready but the pipeline cannot accept it |
| Wait | Thread hit `__syncthreads()` and is blocked waiting for all threads in the block to arrive |

---

**Naive — low stalls, slow from instruction volume:**

Naive's stalls are all low: 7.22% long scoreboard, 0.05% MIO, 5.42% wait. No single
stall dominates. The kernel is slow not because threads are frozen — they are actually
executing most of the time. The problem is that it executes 156.7M instructions to
produce the same result that cuBLAS gets in 42.4M.

The long scoreboard (7.22%) is lower than you might expect for a kernel with no shared
memory. The L1/L2 cache absorbs many of the strided B accesses because warps computing
different rows of C all read the same columns of B — the same data requested by
thousands of threads gets cached and reused, even without explicit tiling.

**Tiled — MIO throttle at 49.73% is the dominant bottleneck:**

The tiled kernel's stall profile is completely different from naive. The inner
dot-product loop:

```cu
for (int k = 0; k < TILE_SIZE; k++)
    sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
```

runs 16 iterations per tile step, and there are 64 tile steps (K / TILE_SIZE).
That is 1024 shared memory reads per thread. All 256 threads in the block issue
shared memory reads every cycle, and the MIO (memory input/output) pipeline — which
serves shared memory — cannot keep up. Threads queue behind it, burning 49.73% of
their active cycles doing nothing.

This is the core limitation of a 16×16 tile: the compute-to-shared-memory-access
ratio is only 1 FMA per shared memory read. There is not enough arithmetic to hide
the MIO pipeline pressure.

The long scoreboard also rises to 18.81% — higher than naive — because global loads
for the next tile happen while threads are already stalled on MIO from the current
tile's dot product. Both pipelines are under pressure simultaneously.

Despite these stalls, tiled is faster than naive (2.15 ms vs 3.51 ms) because it
executes 18% fewer total instructions (129M vs 156M) and threads are stalled for
shorter absolute durations even when the stall percentage is higher.

**cuBLAS — no dominant stall, all three balanced at ~10%:**

cuBLAS has three stall types all near 10%: long scoreboard 9.93%, MIO throttle 9.12%,
wait 12.51%. No single bottleneck dominates. This balanced profile is the signature of
a well-optimised kernel — every source of overhead has been reduced to roughly equal
weight so that eliminating any one of them would give only marginal gain.

The wait stall (12.51%) is higher than in our kernels. This is expected: cuBLAS uses
more `__syncthreads()` barriers per tile because its larger tiles require more careful
double-buffering synchronisation between loading and computing phases.

---

## 4. Occupancy

```
                        Naive    Tiled 16×16    cuBLAS 128×64
Achieved occupancy %  : 98.67      98.66           46.74
```

**cuBLAS has half the occupancy of our kernels and is 6× faster.**

This is the clearest possible demonstration that occupancy is not performance. Our
kernels fill the SM with warps (98.7%) but those warps spend half their time stalled
on MIO or executing redundant instructions. cuBLAS keeps only 46.74% of the warps
resident — but each warp is doing far more useful work per cycle.

cuBLAS uses 128 threads per block (a single warp group), each thread computing 64
output elements accumulated in registers. With fewer but more productive threads, the
SM needs fewer warps to stay busy. Latency is hidden through arithmetic depth — each
thread has so many FMAs in its instruction stream that stalls from one load are covered
by the computation from the previous load's results.

Our kernels rely on **latency hiding via warp switching** — when one warp stalls,
the scheduler switches to another. cuBLAS relies on **latency hiding via instruction-
level parallelism** — the thread itself has enough independent work to keep the
pipeline filled. The cuBLAS approach requires fewer warps and leaves register
resources free, which is why its occupancy is lower.

---

## 5. cuBLAS Internals

The profiler reveals the internal kernel cuBLAS selected:

```
volta_sgemm_128x64_nn  (8, 16, 5) blocks × (128, 1, 1) threads
```

**What the name tells you:**

| Part | Meaning |
|---|---|
| `volta` | Uses the Volta SM SGEMM implementation (T4 is Turing but uses Volta's FP32 path — Turing tensor cores target INT8/FP16, not FP32) |
| `sgemm` | Single-precision (float32) GEMM |
| `128x64` | Output tile per block: 128 rows × 64 columns |
| `nn` | No-transpose A, no-transpose B |

**Grid: (8, 16, 5) = 640 blocks total**

With a 128×64 output tile and a 1024×1024 output matrix:
- Blocks along N: 1024 / 64 = 16 ✓
- Blocks along M: 1024 / 128 = 8 ✓
- The 3rd grid dimension (5) tiles along K for the reduction phase

**Each thread computes 64 output elements:**

128 threads per block × 64 elements per thread = 8192 output elements per block
(128 rows × 64 cols = 8192 ✓). Each thread accumulates 64 partial sums in registers
across K=1024 steps — this register accumulation is what drives the 3× instruction
count reduction vs our tiled kernel (42.4M vs 129.1M).

**DRAM reads: 40 MB vs 165 MB (4× less):**

With 128-column tiles, each element of B is reused by 128 threads instead of 16.
Reuse factor = 128 / 16 = 8×, which approximately matches the 4× reduction in
DRAM reads (the remaining factor is cache effects and alignment overhead).

---

## 6. Key Takeaways

**Tiled's bottleneck is shared memory (MIO), not global memory.**
49.73% of tiled's active cycles are stalled on the shared memory pipeline. The 16×16
tile gives only 1 FMA per shared memory read — not enough arithmetic to keep the MIO
pipeline from saturating. This is the core limitation of small tiles.

**Naive's bottleneck is instruction volume, not memory stalls.**
Despite accessing B in a strided pattern with no explicit reuse, naive's long scoreboard
stall is only 7.22%. The L1/L2 cache provides implicit reuse across warps computing
the same columns of C. The kernel is slow because it executes 156.7M instructions —
3.7× more than cuBLAS for the same result.

**High occupancy masked the real bottleneck in the tiled kernel.**
Both our kernels achieve 98.7% occupancy. Without the stall breakdown, you might
conclude the kernels are well-optimised. The MIO throttle at 49.73% reveals that the
SM is full of warps but those warps are all queued behind the same shared memory
pipeline — adding more warps cannot help.

**cuBLAS uses 4× less DRAM and 3× fewer instructions.**
Both improvements come from the same source: a 128×64 tile where each thread
accumulates 64 outputs in registers. Every byte loaded from global memory is used
for 128 multiply-adds instead of 16. Every shared memory read feeds multiple FMAs
instead of one. The MIO pressure drops from 49.73% to 9.12% as a direct result.

**Occupancy is not performance — cuBLAS at 46.74% beats our 98.67%.**
cuBLAS hides latency through arithmetic depth per thread, not through warp switching.
Fewer warps, each doing more work, leaves register resources available for the large
per-thread accumulator arrays and produces a more balanced stall profile (~10% each)
with no dominant bottleneck.
