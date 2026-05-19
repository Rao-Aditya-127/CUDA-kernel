# Understanding Nsight Compute (NCU) Profiling Output
### A beginner-friendly guide for GPU kernel optimization

---

## The Big Picture: What Are You Even Measuring?

When you run a GPU kernel, there are two things that can slow it down:

1. **Compute bottleneck** — The GPU math units (SMs) are the slow part. Not enough arithmetic is happening.
2. **Memory bottleneck** — The GPU is fast but keeps waiting for data from memory.

Your job as a kernel author is to figure out *which one is the problem* and fix it. NCU gives you the data to answer that question.

---

## Section 1: GPU Speed of Light — Start Here Every Time

This is the first section in NCU output and the first thing you should look at. It tells you how close you are to the hardware's maximum capability.

```
Memory Throughput      %        60.96   ← naive
Memory Throughput      %        73.93   ← tiled

Compute (SM) Throughput  %      60.96   ← naive
Compute (SM) Throughput  %      73.93   ← tiled
```

**What these mean:**

- `Compute (SM) Throughput %` — What fraction of the GPU's maximum compute capacity you are using. 100% = fully saturated.
- `Memory Throughput %` — What fraction of the GPU's maximum memory bandwidth you are using. 100% = fully saturated.

**The key insight:** Look at which one is higher. That is your bottleneck.

In your case, both numbers are equal for both kernels. NCU itself confirms this:

> *"Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced."*

This means neither compute nor memory is clearly the bottleneck — they are both underperforming together. The problem is elsewhere (spoiler: warps are stalling, explained below).

**Your numbers at a glance:**

| Kernel | Compute % | Memory % | Duration |
|--------|-----------|----------|----------|
| Naive  | 60.96     | 60.96    | 3.52 ms  |
| Tiled  | 73.93     | 73.93    | 2.18 ms  |

Tiled is clearly better — higher utilization, shorter duration.

---

## Section 2: Roofline — How Far Are You From Peak?

```
Naive  achieved 8%  of FP32 peak performance
Tiled  achieved 12% of FP32 peak performance
```

**What this means:**

The T4 has a theoretical FP32 peak of ~8.1 TFLOPS. Your naive kernel is using only 8% of that. Your tiled kernel improved to 12%, which is better but still far from peak.

**Why is this useful?**

It tells you the ceiling for improvement. If you were at 80%, there's little headroom. At 8-12%, you have enormous room to optimize — which is expected for early kernels.

**What to track over time:** This number should grow as you improve your kernels. A well-optimized matmul should get to 60-80%+ of peak FP32.

---

## Section 3: Scheduler Statistics — The Most Important Section for Beginners

This section tells you how busy the GPU instruction schedulers are. Each SM has 4 schedulers, and each can issue one instruction per cycle. If they are idle, your kernel is wasting cycles.

```
                          Naive    Tiled
One or More Eligible %    17.89    23.78   ← only 18-24% of cycles have ready work
No Eligible %             82.11    76.22   ← 76-82% of cycles are idle!
Issued Warp Per Scheduler  0.18     0.24
```

**Plain English:**

- The scheduler is idle 82% of the time in the naive kernel.
- Even in the tiled kernel, it is idle 76% of the time.
- This means 3 out of every 4 cycles, the GPU is doing nothing useful.

**Why does this happen?**

Warps are *waiting* — either for memory to come back, or for the instruction queue to clear. The GPU is not being kept busy. This is the core problem in both your kernels.

**Active Warps vs Eligible Warps:**

```
Active Warps Per Scheduler    7.90   (warps that exist and are tracked)
Eligible Warps Per Scheduler  0.72   (warps that are READY to issue an instruction)
```

You have ~8 active warps per scheduler, but only 0.72 are ready at any given cycle. Most warps are stalled, waiting for something.

**What to improve:** You want `One or More Eligible %` to be as high as possible. Good kernels often reach 60-80%+.

---

## Section 4: Warp State Statistics — Why Are Warps Stalling?

This tells you *why* warps are waiting. This is where naive and tiled diverge.

### Naive Kernel:

```
Warp Cycles Per Issued Instruction   44.15
```

On average, a warp waits 44 cycles between issuing instructions. That is a lot of wasted time.

**The dominant stall reason:**

> *"Each warp spends 34.3 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full."*

Translation: The naive kernel fires so many global memory requests (reads from A and B matrices directly from DRAM/L2 every time) that the memory instruction queue fills up and warps have to wait for it to drain.

This is the classic symptom of **uncoalesced or repeated global memory access**.

### Tiled Kernel:

```
Warp Cycles Per Issued Instruction   33.19
```

Down from 44 to 33 cycles — better! But still high.

**The dominant stall reason:**

> *"Each warp spends 16.7 cycles being stalled waiting for the MIO (memory input/output) instruction queue."*

The stall moved from global memory (LG queue) to shared memory (MIO queue). This is actually *progress* — you successfully moved the bottleneck from slow global memory to fast shared memory. But shared memory accesses are now the new bottleneck.

**What this tells you about tiling:** Tiling worked — it reduced global memory pressure. The next step is to reduce shared memory pressure (e.g., avoid bank conflicts, use wider loads).

---

## Section 5: Memory Workload Analysis — How Is Memory Being Used?

### L1/TEX Hit Rate:

```
Naive:  87.36%   ← high L1 hit rate
Tiled:   6.51%   ← very low L1 hit rate
```

This seems backwards at first — why does tiled have a *worse* L1 hit rate?

Because the tiled kernel deliberately loads data into **shared memory** first, then reads from shared memory. So it bypasses L1 cache (shared memory is separate). The low L1 hit rate in tiled is expected and fine — it means data is coming from shared memory instead.

### L2 Hit Rate:

```
Naive:  81.68%
Tiled:  83.40%
```

Both are similar and healthy. L2 is serving most requests that miss L1.

### DRAM Throughput:

```
Naive:  17.46%  (of max DRAM bandwidth)
Tiled:  24.40%
```

Tiled actually uses *more* DRAM bandwidth — because it is doing more work per unit time (higher throughput). This is a good sign. You want DRAM to be busy, just not the bottleneck.

### The Uncoalesced Access Warning (Naive kernel):

> *"Only 18.0 of the 32 bytes transmitted per sector are utilized by each thread. This could be caused by a stride between threads."*

This is a key optimization opportunity for the naive kernel. When threads in a warp read memory, they should read consecutive addresses (coalesced). If thread 0 reads address 0 and thread 1 reads address 1024 (strided), the GPU has to issue many separate memory transactions instead of one wide transaction.

In naive matmul, accessing matrix B column-wise causes exactly this strided pattern.

---

## Section 6: Occupancy — Are You Filling the GPU?

```
Theoretical Occupancy    %    100     (both kernels)
Achieved Occupancy       %    98.74   (naive)
Achieved Occupancy       %    98.68   (tiled)
```

**Occupancy** = what fraction of the maximum possible warps are active on the GPU at once.

Both kernels achieve near-perfect occupancy (98%+). This is good — it means you are not wasting hardware slots. The GPU has plenty of warps to switch between.

**Important insight:** High occupancy does NOT mean good performance. Your kernels have 98% occupancy but only 8-12% peak compute utilization. Occupancy is a prerequisite for hiding latency, but it does not guarantee that the warps are doing useful work.

**What limits occupancy:** 

- Registers per thread (naive uses 52, tiled uses 39)
- Shared memory per block (tiled uses 2.05 KB/block)
- Block size

For tiled, `Block Limit Registers = 6` and `Block Limit Warps = 4` — the warp limit is the binding constraint. Both are still at 100% theoretical, so you are fine here.

---

## Section 7: Launch Statistics — Grid and Block Configuration

```
Block Size    256   (16x16 threads)
Grid Size    4096   (64x64 blocks)
```

For a 1024x1024 matrix with 16x16 blocks: 1024/16 = 64 blocks per dimension, so 64x64 = 4096 total blocks. This is correct.

**Registers Per Thread:**

```
Naive:  52 registers/thread
Tiled:  39 registers/thread
```

The tiled kernel uses fewer registers per thread because it reuses shared memory more efficiently. Fewer registers = better occupancy potential.

**Shared Memory Per Block:**

```
Naive:  0 KB    (no shared memory used)
Tiled:  2.05 KB (two 16x16 float tiles = 2 * 16 * 16 * 4 bytes = 2048 bytes ≈ 2 KB)
```

The tiled kernel's shared memory usage is exactly what you expect from a 16x16 tile of floats × 2 matrices.

---

## What to Track — Your Optimization Scorecard

Every time you write or optimize a kernel, record these metrics. They tell the full story.

| Metric | Where in NCU | What Good Looks Like | Your Naive | Your Tiled |
|--------|-------------|----------------------|------------|------------|
| **Duration (ms)** | Speed of Light | As low as possible | 3.52 ms | 2.18 ms |
| **FP32 Peak %** | Roofline | > 50% for matmul | 8% | 12% |
| **Compute Throughput %** | Speed of Light | High, balanced with memory | 60.96% | 73.93% |
| **Memory Throughput %** | Speed of Light | High, balanced with compute | 60.96% | 73.93% |
| **Scheduler Eligible %** | Scheduler Stats | > 60% | 17.89% | 23.78% |
| **Warp Cycles Per Instruction** | Warp State | < 10 ideally | 44.15 | 33.19 |
| **Achieved Occupancy %** | Occupancy | > 75% | 98.74% | 98.68% |
| **L2 Hit Rate %** | Memory Workload | > 80% | 81.68% | 83.40% |
| **DRAM Throughput %** | Speed of Light | Should not be 100% (bottleneck) | 17.46% | 24.40% |

---

## Reading the Output: A 5-Step Process

Every time you profile a new kernel, go through these steps in order:

**Step 1 — Check Speed of Light**  
Are compute and memory throughput balanced? Which one is higher = your bottleneck.

**Step 2 — Check Roofline**  
What % of FP32 peak are you achieving? This gives you a single progress number.

**Step 3 — Check Scheduler Stats**  
What % of cycles have eligible warps? If "No Eligible %" > 50%, warps are stalling badly.

**Step 4 — Check Warp State Stats**  
What are warps waiting for? (LG queue = global memory problem, MIO queue = shared memory problem, barrier = synchronization problem)

**Step 5 — Check Memory Workload**  
Are there uncoalesced accesses? What are L1/L2 hit rates? Is DRAM the bottleneck?

---

## What to Work on Next (Based on Your Results)

Your tiled kernel improved over naive but both still have significant headroom. Here is a prioritized list:

**1. Fix the MIO stall in tiled (biggest gain)**  
The tiled kernel stalls on shared memory instruction queue. Solutions:
- Use `float4` vectorized loads to load 4 floats per instruction instead of 1
- Check for shared memory bank conflicts (32 banks, stride-1 access is ideal)

**2. Fix uncoalesced global loads in naive**  
Only 18/32 bytes per sector are used. Accessing matrix B by column causes stride-32 access pattern. This is inherent to naive matmul — tiling solves it by loading B tiles row-wise into shared memory.

**3. Increase compute intensity**  
Both kernels are at 8-12% FP32 peak. The next big jump comes from:
- Larger tiles (32x32 instead of 16x16) — more reuse per global load
- Register blocking / thread coarsening — each thread computes a 2x2 or 4x4 output tile
- Tensor cores (Wmma API) — available on T4 (sm_75), gives 8x peak FP16 throughput

**4. Reduce warp cycles per instruction below 20**  
Current: 33-44 cycles. Target: < 20 cycles. This requires reducing stalls from all sources above.

---

## Quick Glossary

| Term | Plain English |
|------|--------------|
| SM (Streaming Multiprocessor) | One "core complex" on the GPU. T4 has 40 SMs. |
| Warp | 32 threads that execute together in lockstep. |
| Occupancy | What % of maximum possible warps are active. |
| Coalescing | Threads in a warp reading consecutive memory addresses — efficient. |
| Shared Memory | Fast on-chip memory per SM (~48 KB). Manual L1 cache. |
| L1/TEX Cache | Automatic hardware cache, closest to SM. |
| L2 Cache | Shared cache across all SMs. |
| DRAM / HBM | Off-chip GPU memory. Slowest but largest (16 GB on T4). |
| LG Queue | Instruction queue for local/global memory operations. |
| MIO Queue | Instruction queue for shared memory + special math operations. |
| Roofline | Chart showing if you are compute-bound or memory-bound vs hardware limits. |
| IPC (Instructions Per Cycle) | How many instructions execute per clock cycle. Higher = better. |
| Bank Conflict | Two threads in a warp accessing same shared memory bank — serialized. |