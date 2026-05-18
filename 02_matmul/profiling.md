# Matrix Multiplication — Naive vs Tiled Kernel Profiling

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA T4 (CC 7.5) |
| SMs | 40 |
| CUDA | sm_75 |
| Matrix size | 1024 × 1024 × 1024 (M=K=N=1024) |
| Block size | 16 × 16 = 256 threads |
| Grid size | 64 × 64 = 4096 blocks |
| Tile size | 16 × 16 |
| Compile flags | `-O2 -arch=sm_75 -lineinfo` |

---

## Results Summary

| Metric | Naive | Tiled | Improvement |
|---|---|---|---|
| **Kernel duration** | 3.52 ms | 2.18 ms | **1.61× faster** |
| **Elapsed cycles** | 5,535,558 | 3,421,653 | 38% fewer cycles |
| **Memory throughput** | 55.20 GB/s | 77.33 GB/s | 40% higher |
| **Compute (SM) throughput** | 60.96% | 73.93% | +13 pp |
| **DRAM throughput** | 17.46% | 24.40% | +7 pp |
| **L1/TEX hit rate** | 87.36% | 6.51% | — (see note) |
| **L2 hit rate** | 81.68% | 83.40% | similar |
| **Warp cycles / instruction** | 44.15 | 33.19 | 25% fewer stall cycles |
| **Issue slots busy** | 17.76% | 23.65% | +6 pp |
| **FP32 peak utilisation** | 8% | 12% | 1.5× |
| **Achieved occupancy** | 98.74% | 98.68% | equal |
| **Registers per thread** | 52 | 39 | 25% fewer |
| **Static shared mem / block** | 0 KB | 2.05 KB | tiled uses SRAM |
| **Executed instructions** | 156,631,040 | 129,040,384 | 18% fewer |

---

## 1. Execution Time

```
Naive :  3.643 ms   (CUDA event timing, includes kernel launch overhead)
Tiled :  2.196 ms

Speedup = 3.643 / 2.196 = 1.66×

ncu reported durations (pure kernel time):
Naive :  3.52 ms
Tiled :  2.18 ms
Speedup = 3.52 / 2.18 = 1.61×
```

The tiled kernel is ~1.6× faster on a 1024×1024 matrix with TILE_SIZE=16.
The gap widens with larger matrices because the benefit of reuse scales
with K — at K=1024 each tile is reused 16 times (K / TILE_SIZE = 64 tiles,
each giving 16 multiply-adds from shared memory instead of global memory).

---

## 2. Memory Access Pattern

This is the core story of the optimisation.

### Naive kernel — hammering global memory

```
L1/TEX Hit Rate  :  87.36%
L2 Hit Rate      :  81.68%
DRAM Throughput  :  17.46%
Memory Throughput:  55.20 GB/s
```

The high L1 hit rate (87%) looks good on the surface, but ncu flags it
with a warning:

> "On average, only 18.0 of the 32 bytes transmitted per sector are
> utilized by each thread. This could possibly be caused by a stride
> between threads."

This is the **uncoalesced access problem**. When threads in a warp read
column `col` of matrix B — `B[k * N + col]` — each thread reads a
different row of B, which means accesses are strided by N=1024 floats
(4096 bytes). A 128-byte cache line is fetched but only 4 bytes are used.
This wastes 87.5% of every cache line fetched for B.

The L1 cache absorbs most of these misses (87% hit rate) because threads
in different warps computing different rows of C reuse the same columns of
B. But this causes L1 contention, which is why the LG (local/global)
queue stall dominates.

### Tiled kernel — shared memory changes everything

```
L1/TEX Hit Rate  :  6.51%
L2 Hit Rate      :  83.40%
DRAM Throughput  :  24.40%
Memory Throughput:  77.33 GB/s
Static Shared Mem:  2.05 KB/block
```

The L1 hit rate **drops to 6.51%** in the tiled kernel. This is not a
regression — it is expected and correct. The tiled kernel deliberately
bypasses L1 for global memory loads (it loads each tile once into shared
memory), so there are far fewer L1 requests to begin with. Shared memory
accesses do not show up in the L1 hit rate counter.

The tiled kernel loads each 16×16 tile of A and B from global memory
exactly once per tile iteration. Those 256 values are then read 16 times
each from shared memory during the compute phase. This gives:

```
Global memory reads per output element:
  Naive :  2 × K     = 2048 global reads
  Tiled :  2 × K / TILE_SIZE = 128 global reads

Reduction = 16× fewer global memory reads
```

The higher DRAM throughput (24.40% vs 17.46%) means the tiled kernel
actually **uses DRAM more efficiently** — when it does go to DRAM, it
accesses memory in coalesced tile-shaped patterns rather than scattered
column accesses.

---

## 3. Warp Stall Analysis

Warp stalls tell you what threads are waiting for. This is the clearest
signal of where time is being lost.

### Naive kernel — stalled on global memory queue

```
Warp Cycles Per Issued Instruction  :  44.15 cycles
No Eligible warps                   :  82.11% of cycles
One or More Eligible                :  17.89% of cycles
```

ncu identifies the primary stall reason:

> "Each warp spends 34.3 cycles stalled waiting for the L1 instruction
> queue for local and global (LG) memory operations to be not full.
> This stall type represents 77.6% of the total 44.1 cycles between
> issuing two instructions."

In plain terms: the global memory pipeline is so congested that warps
queue up waiting to issue their loads. Even with 7.90 active warps per
scheduler (near the 8-warp maximum), only 0.72 warps are eligible per
cycle because the rest are stuck in the LG queue. The GPU's latency
hiding mechanism (switching between warps) is partially working — it
keeps SM active cycles high — but the pipeline is fundamentally bottlenecked
on how fast it can issue global memory requests.

### Tiled kernel — stalled on shared memory pipeline

```
Warp Cycles Per Issued Instruction  :  33.19 cycles  (25% better)
No Eligible warps                   :  76.22% of cycles
One or More Eligible                :  23.78% of cycles
```

The dominant stall reason changes completely:

> "Each warp spends 16.7 cycles stalled waiting for the MIO
> (memory input/output) instruction queue. This stall reason is high
> in cases of extreme utilisation of the MIO pipelines, which include
> shared memory instructions."

The tiled kernel's stalls are now caused by shared memory throughput
pressure, not global memory latency. Shared memory latency is ~5 cycles
vs ~600 cycles for global memory. The total stall cycles drop from 44.15
to 33.19 — a 25% reduction — but there is still room to improve further
(bank conflict analysis, wider loads, double buffering).

The instruction issue rate improves from 17.89% to 23.78% eligible cycles,
meaning the scheduler finds a ready warp to issue more often.

---

## 4. Compute Efficiency

```
                    Naive     Tiled
SM Busy           : 20.34%   26.73%
Issue Slots Busy  : 17.76%   23.65%
Executed IPC      :  0.72     0.95
FP32 Peak Used    :  8%       12%
```

Both kernels are well below peak compute — the T4 peak is ~8.1 TFLOPS
and a 1024³ matmul is only ~2.1 GFLOP, so we are not compute bound at
this size. The tiled kernel executes more instructions per cycle (IPC
0.95 vs 0.72) because warps spend less time stalled, so the scheduler
finds eligible warps more often.

The roofline position: both kernels sit in the memory-bound region, but
tiled moves further right (higher arithmetic intensity) and further up
(higher throughput) compared to naive.

---

## 5. Occupancy

```
                           Naive    Tiled
Theoretical Occupancy    :  100%    100%
Achieved Occupancy       : 98.74%  98.68%
Active Warps Per SM      : 31.60   31.58
Block Limit (Registers)  :   4       6
Registers Per Thread     :  52      39
```

Both kernels achieve near-perfect occupancy (~98.7%), meaning the SM is
always holding close to its maximum 32 warps. This rules out occupancy
as the bottleneck for either kernel.

One interesting difference: the naive kernel uses 52 registers per thread
vs 39 for the tiled kernel. This is counter-intuitive — you might expect
shared memory usage to increase register pressure — but the tiled kernel
is more structured and the compiler can optimise register allocation
better. The lower register count in the tiled kernel also explains why
its block limit from registers is 6 (vs 4 for naive), giving more
scheduling flexibility.

The naive kernel has 0 bytes of static shared memory per block, confirming
it reads everything directly from global memory. The tiled kernel allocates
2.05 KB per block (two 16×16 float tiles = 2 × 16 × 16 × 4 = 2048 bytes).

---

## 6. Instruction Count

```
Naive :  156,631,040 instructions
Tiled :  129,040,384 instructions

Reduction = 17.6% fewer instructions
```

The tiled kernel executes 17.6% fewer total instructions. This comes from
two sources. First, fewer global memory load instructions because tiles
amortise the loads across 16 threads. Second, 39 registers per thread vs
52 means fewer register spill/restore instructions (though neither kernel
spills — confirmed by `Local Memory Spilling Requests = 0` for both).

---

## 7. Key Takeaways

**The 1.6× speedup comes entirely from reducing global memory pressure.**
The tiled kernel issues 16× fewer global memory reads per output element
by staging data through shared memory. Every other metric flows from this
single change.

**The stall reason shifts from LG queue (global memory) to MIO queue
(shared memory).** This is the expected signature of a correct tiling
optimisation. Shared memory stalls at 5-cycle latency are far cheaper
than global memory stalls at 600-cycle latency.

**Occupancy is not the bottleneck — both kernels hit 98.7%.** The GPU
is always full of warps. The problem is that in the naive kernel, most
of those warps are stuck waiting for global memory, not doing useful work.

**Both kernels are still far from peak FP32 performance (8% and 12%).**
The T4 peak is 8.1 TFLOPS. A 1024³ matmul at 16×16 tiles is still
memory-bound. To approach peak compute you would need larger tiles
(32×32), vectorised loads (float4), double buffering to overlap load and
compute phases, or tensor core instructions (wmma).

**The tiled kernel uses 25% fewer registers per thread (39 vs 52).**
This is a compiler artefact of more structured code, not a deliberate
optimisation, but it gives the tiled kernel a higher theoretical block
limit from the register perspective (6 blocks vs 4).

---

## 8. What to Try Next

| Optimisation | Expected gain | Why |
|---|---|---|
| Increase TILE_SIZE to 32 | 2× more reuse per global load | More arithmetic per byte fetched |
| Use `float4` vectorised loads | Better memory bandwidth utilisation | 128-bit loads instead of 32-bit |
| Double buffering (`__pipeline`) | Overlap load and compute | Hide shared memory latency |
| Tensor cores (`wmma`) | 8× compute throughput | Hardware accelerated 16×16 matmul |
| Transposed tile_B layout | Eliminate bank conflicts | Column access → row access in SRAM |