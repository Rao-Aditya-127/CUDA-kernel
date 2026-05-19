# Reduction — Kernel Profiling

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA T4 (CC 7.5) |
| Array size | 16,777,216 floats (~64 MB) |
| Block size | 256 threads |
| Grid size | 65,536 blocks (our kernels) / 800 blocks (CUB) |
| Compile flags | `-O2 -arch=sm_89 -lineinfo` |

---

## Results Summary

```
Naive  :  482.134 ms   result = 16777216   PASS
Shared :    0.676 ms   result = 16777216   PASS
Warp   :    1.080 ms   result = 16777216   PASS
WarpBlk:    0.422 ms   result = 16777216   PASS
CUB    :    0.284 ms   result = 16777216   PASS
```

| Metric | Shared | Warp | Warp Block | CUB |
|---|---|---|---|---|
| **Duration — CUDA event (ms)** | 0.676 | 1.080 | 0.422 | 0.284 |
| **Duration — ncu (ms)** | 0.626 | 1.040 | 0.364 | 0.255 |
| **Speedup vs Shared** | 1.00× | 0.63× *(slower)* | **1.72×** | **2.45×** |
| **DRAM throughput %** | 41.00 | 24.45 | 65.95 | 97.95 |
| **SM throughput %** | 72.40 | 12.62 | 40.27 | 2.41 |
| **DRAM bytes read (MB)** | 80.58 | 79.99 | 75.18 | 77.00 |
| **Achieved occupancy %** | 88.98 | 93.02 | 74.90 | 98.72 |
| **Total instructions** | 52,166,656 | 14,680,064 | 16,908,288 | — |
| **Stall — long scoreboard %** | 23.13 | **90.17** | 53.26 | — |
| **Stall — wait % (`__syncthreads`)** | 19.86 | 2.22 | 10.96 | — |
| **Stall — MIO throttle %** | 7.86 | 0.14 | 1.54 | — |
| **Shared mem stores (instructions)** | 4,718,592 | 0 | 524,288 | — |
| **Shared mem loads (instructions)** | 8,454,144 | 0 | 65,536 | — |

---

## 1. Execution Time

```
Naive  :  482.134 ms   (one thread, 16M serial additions — GPU completely wasted)
Shared :    0.676 ms
Warp   :    1.080 ms   ← slower than Shared despite using faster shuffle instructions
WarpBlk:    0.422 ms
CUB    :    0.284 ms

Naive  vs CUB    :  1697× slower
Warp   vs Shared :  0.63× — Warp is 60% slower than Shared
WarpBlk vs Warp  :  2.56× speedup  (same shuffle instructions, 8× fewer atomics)
CUB    vs WarpBlk:  1.49× speedup
```

**The headline result: Warp is slower than Shared.**

This is the atomic contention problem made visible. Warp eliminates `__syncthreads()`
and uses faster shuffle instructions — yet it is still slower. The reason is that
`reduce_warp` issues 8 `atomicAdd` calls per block (one per warp), giving
524,288 total atomics across 65,536 blocks. All of them target the same single
address in global memory. They queue up and serialise at the L2 cache, wiping out
the gain from faster intra-block reduction.

`reduce_warp_block` fixes this with one extra step: after each warp reduces to its
lane 0, those 8 partial sums are written into shared memory and the first warp
reduces them with one more shuffle. Now only thread 0 calls `atomicAdd` — back to
65,536 atomics total, matching Shared — and the shuffle speed advantage is
fully realised: 2.56× faster than plain Warp.

CUB's remaining 1.49× advantage over Warp Block comes from processing multiple
elements per thread before entering the reduction, which raises arithmetic intensity
and keeps the memory pipeline busier.

---

## 2. Memory Bandwidth

```
                    Shared    Warp     Warp Block    CUB
DRAM throughput % :  41.00    24.45      65.95       97.95
DRAM bytes read MB:  80.58    79.99      75.18       77.00
SM throughput %   :  72.40    12.62      40.27        2.41
```

All four parallel kernels do the same job — read ~64 MB of floats once.
Their DRAM bytes read are nearly identical (~77–81 MB; the small variation is
read-amplification from partial cache-line fetches at block boundaries).

**DRAM throughput %** is the fraction of the GPU's peak memory bandwidth actually
used. The higher this is, the more efficiently the kernel is driving the memory bus.

**The Warp anomaly — high occupancy, low DRAM throughput:**

```
Warp occupancy    :  93.02%   (SMs are full of active warps)
Warp DRAM tput    :  24.45%   (memory bus is mostly idle)
Warp SM tput      :  12.62%   (SMs are barely doing work)
```

Warp has the highest occupancy of any parallel kernel — 93% of the maximum
possible warps are resident on the SM. Yet the memory bus is only 24.45% utilized
and the SM is only 12.62% active. This contradiction has one explanation:
**the warps are present but stuck, all queued waiting for their `atomicAdd` to
complete**. They are counted as "active" in occupancy but are not issuing memory
loads or doing compute.

This is the most important thing the profiler reveals: **occupancy is not
performance**. A GPU full of stalled warps is no better than an empty one.

**Warp Block fixes it:**

Reducing atomics from 524,288 to 65,536 lets warps spend their time issuing
memory loads instead of waiting. DRAM throughput jumps from 24.45% → 65.95%,
SM throughput from 12.62% → 40.27%, and duration drops by 2.86×.

**CUB nearly saturates the memory bus (97.95%):**

CUB uses only 800 blocks. Each thread loads ~82 elements via a grid-stride loop
before reducing — so the kernel spends almost all of its time fetching data, not
in reduction logic. The SM throughput (2.41%) looks low, but this is expected:
CUB is a pure memory-bound kernel where SMs issue loads continuously and the
compute (addition) is nearly free compared to the memory wait time.

---

## 3. Shared Memory Usage

*(l2__global_atomic_store_bytes not available on CC 7.5 — atomic counts derived from code)*

```
                              Shared       Warp      Warp Block
atomicAdd calls (total)   :   65,536     524,288       65,536
Shared mem store instrs   :4,718,592           0      524,288
Shared mem load instrs    :8,454,144           0       65,536
Total shared mem instrs   :13,172,736          0      589,824   (22× fewer than Shared)
```

**Atomic counts — derived from code, not profiler:**

Every `atomicAdd` to the global output serialises at the L2 cache — only one can
complete at a time on the same address.

- **Shared**: 1 `atomicAdd` per block × 65,536 blocks = 65,536
- **Warp**: 8 per block (one per warp) × 65,536 = 524,288 — 8× more contention
- **Warp Block**: collects 8 warp sums in shared memory and reduces them with the
  first warp before calling `atomicAdd` once — back to 65,536

The 90.17% long scoreboard stall on Warp (vs 23.13% on Shared) is the profiler
showing this 8× difference in atomic pressure.

**Shared memory instruction counts — what the code actually does:**

**Warp** has exactly 0 shared memory instructions. Warp shuffles operate entirely
on registers — no shared memory involved at any step.

**Warp Block** shared memory use is minimal and perfectly accounted for:
- **524,288 stores** = 8 warps per block × lane-0 only writes `warp_sums[]` × 65,536 blocks
- **65,536 loads** = 1 warp-level load (first warp reads all 8 slots in one instruction) × 65,536 blocks

**Shared** carries 13.2M shared memory instructions — 22× more than Warp Block.
This comes from 65,536 blocks each running 8 tree-reduction steps, where every
active thread reads and writes shared memory once per step. The tree loop accesses
shared memory (128+64+32+16+8+4+2+1) × 2 = 510 times per block for reads alone.
This is what the 7.86% MIO throttle stall reflects — the shared memory pipeline
is under sustained pressure throughout the kernel.

---

## 4. Warp Stall Analysis

```
                               Shared    Warp      Warp Block
Long scoreboard stall %      :  23.13    90.17       53.26
Wait stall % (__syncthreads) :  19.86     2.22       10.96
MIO throttle stall %         :   7.86     0.14        1.54
Total instructions            : 52.2M    14.7M       16.9M
```

*(smsp__warp_cycles_per_issued_instruction not available on CC 7.5)*

**What each stall type means:**

| Stall | Cause |
|---|---|
| Long scoreboard | Thread issued a global load or `atomicAdd` and is waiting for the result to return from L2/DRAM |
| Wait | Thread reached `__syncthreads()` and is blocked until all other threads in the block catch up |
| MIO throttle | The shared memory or shuffle instruction pipeline is saturated; thread has an instruction ready but cannot issue it |

---

**Warp — long scoreboard 90.17%**

Nine out of every ten cycles, every thread in every warp is frozen waiting for
an `atomicAdd` to return. Warp has 524,288 atomics all targeting one L2 cache
line — they serialise into a queue, and each warp just sits idle until its turn.

The instruction count makes this concrete: Warp executes only 14.7M instructions
— 3.55× fewer than Shared — yet it is 1.66× *slower*. Each instruction takes
much longer to complete because the threads spend 90% of their time not executing
anything at all.

**Shared — long scoreboard 23.13%, wait 19.86%**

Shared carries two separate stall costs:
- **23.13% long scoreboard**: 65,536 atomicAdds still create some L2 queue pressure,
  but 8× fewer than Warp so the queue is much shorter.
- **19.86% wait**: `__syncthreads()` fires after every one of the 8 tree reduction
  steps. Faster threads reach the barrier first and idle until the last thread catches
  up. This is real wasted time — it shows up clearly in the 52.2M instruction count
  (most of those extra instructions are threads executing in steps where they are
  not the "active half", still burning instruction slots on the `if (tid < s)` check).

**Warp Block — long scoreboard 53.26%, wait 10.96%**

Warp Block's long scoreboard (53.26%) is higher than Shared's (23.13%) despite
having the same 65,536 atomic count. The reason: Warp Block finishes its
intra-block work much faster (no 8-step tree loop), so all 65,536 blocks reach
their `atomicAdd` nearly simultaneously — creating a tighter burst of contention.
Shared's blocks arrive more spread out in time because the tree reduction takes
longer, naturally staggering the atomics.

Despite 53% long scoreboard stalls, Warp Block is still 1.72× faster than Shared
because it executes only 16.9M total instructions vs 52.2M — the stalls are
shorter in absolute time even when they are a larger fraction.

The 10.96% wait stall comes from the one `__syncthreads()` between writing
`warp_sums[]` and reading it in the first warp.

---

## 5. Occupancy

```
                        Shared    Warp     Warp Block    CUB (main kernel)
Achieved occupancy %  :  88.98    93.02      74.90          98.72
```

*(sm__registers_per_thread_allocated not available on CC 7.5)*

**The counterintuitive result: Warp Block has the lowest occupancy yet is the
fastest hand-written kernel.**

```
Warp       : 93.02% occupancy → 1.040 ms  (slow — warps stalled on atomics)
Warp Block : 74.90% occupancy → 0.364 ms  (fast — fewer warps, all productive)
```

Warp Block uses a small shared memory buffer (`warp_sums[8]` = 32 bytes per
block). This is too small to limit occupancy on its own, but combined with
slightly higher register usage per thread for managing the two-phase reduction,
the compiler packs slightly fewer warps per SM — hence 74.90% vs 93%.

The lower occupancy does not hurt because the bottleneck was never "too few warps."
It was the L2 atomic queue. Once that is fixed, fewer warps that are fully
productive outperform more warps that are mostly stalled.

**CUB occupancy (98.72%) matches its DRAM throughput (97.95%)** — this is the
ideal state. Almost every warp is active, and almost every active warp is issuing
memory loads. There is no slack.

---

## 6. Key Takeaways

**Naive is not a GPU kernel — it is a CPU kernel running on GPU hardware.**
373 ms vs 0.626 ms for Shared = 596× slower. 0.12% DRAM throughput means the
memory bus is essentially idle. One thread doing 16M serial additions cannot
drive any GPU pipeline.

**Occupancy is not performance — Warp proves it.**
Warp has the highest occupancy of any parallel kernel (93.02%) but is the
slowest (1.04 ms). Its warps are all resident but all stalled at the L2 atomic
queue. High occupancy is a necessary condition for performance, not a sufficient one.
The question is what those warps are doing while they're resident.

**Atomic contention is a hidden serialisation point.**
524,288 `atomicAdd` calls all targeting one address create a queue at L2.
Warp's long scoreboard stall hits 90.17% — nine out of ten cycles, every thread
is frozen waiting for an atomic to return. Warp executes 3.55× fewer instructions
than Shared yet is 1.66× slower: each instruction costs far more wall-clock time
because threads spend almost no time actually executing. Warp Block restores
65,536 atomics and the stall drops to 53.26% — a 2.86× speedup from this one change.

**The ceiling is DRAM bandwidth, and CUB nearly hits it (97.95%).**
All parallel kernels read ~77–81 MB at the same rate from DRAM. The difference
is how much time they waste between memory loads. CUB eliminates that waste by
giving each thread 82 elements to load sequentially before reducing — the GPU
spends almost all its time streaming data, not waiting on atomics or barriers.

**CUB uses two kernels, not one.**
The main kernel (800 blocks, 255 µs) handles the 64 MB array. A second tiny
kernel (1 block, 3 µs) reduces the 800 partial sums to a final answer. This
two-level design keeps the main kernel's atomic count at just 800, far below
the 65,536 of our hand-written kernels.

---

## 7. What to Try Next

| Optimisation | What to measure | Why |
|---|---|---|
| Increase elements per thread (grid-stride loop) | DRAM throughput % | Amortises kernel launch overhead; matches CUB's strategy |
| Use `float4` vectorised loads | DRAM bytes read / transaction | 128-bit loads reduce transaction count and improve throughput |
| Reduce to `double` and compare | Duration, DRAM throughput | FP64 is often rate-limited differently; see if bandwidth changes |
| Profile with `nsys` timeline | Kernel gaps, CPU↔GPU overlap | Reveal whether launch overhead or CPU post-processing dominates |
