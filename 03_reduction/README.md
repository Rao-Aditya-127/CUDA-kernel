# Module 03 — Reduction

## What is a Reduction?

A reduction takes an array of N values and collapses it into a single value
by repeatedly applying one operation.

```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]

Sum reduction:  3+1+4+1+5+9+2+6 = 31
Max reduction:  max(3,1,4,1,5,9,2,6) = 9
Min reduction:  min(3,1,4,1,5,9,2,6) = 1
```

You already use reductions constantly in Python without thinking about it:

```python
torch.sum(x)    # sum reduction
torch.max(x)    # max reduction
torch.mean(x)   # sum reduction + divide
torch.norm(x)   # square + sum + sqrt
```

Every one of those eventually calls a GPU reduction kernel.

---

## Why is Parallel Reduction Hard?

On a CPU, summing an array is trivial — one loop:

```python
total = 0
for x in array:
    total += x
```

On a GPU you have thousands of threads. The naive idea would be:
"Let every thread add one element."
But they all want to write to the same output variable at the same time.
That is a **data race** — results would be wrong or random.

The solution is to structure the work so threads never conflict,
and then carefully combine partial results.

---

## Approach 1 — Naive (single thread)

```
Threads:  T0 does everything. T1...T255 do nothing.

T0: reads A[0], A[1], A[2] ... A[N-1] one by one
    accumulates into a local sum
    writes to output
```

This is correct but wastes the entire GPU. It is only useful as a
correctness baseline.

---

## Approach 2 — Shared Memory Tree Reduction

The key insight: **reduction can be parallelised as a tree**.

Imagine 8 values and 8 threads:

```
Initial values in shared memory:
  sdata: [3] [1] [4] [1] [5] [9] [2] [6]
  index:  0   1   2   3   4   5   6   7

Step 1 (s=4): threads 0-3 add from index+4
  T0: sdata[0] += sdata[4]  →  3+5 = 8
  T1: sdata[1] += sdata[5]  →  1+9 = 10
  T2: sdata[2] += sdata[6]  →  4+2 = 6
  T3: sdata[3] += sdata[7]  →  1+6 = 7
  sdata: [8] [10] [6] [7] [5] [9] [2] [6]

Step 2 (s=2): threads 0-1 add from index+2
  T0: sdata[0] += sdata[2]  →  8+6 = 14
  T1: sdata[1] += sdata[3]  →  10+7 = 17
  sdata: [14] [17] [6] [7] ...

Step 3 (s=1): thread 0 adds from index+1
  T0: sdata[0] += sdata[1]  →  14+17 = 31
  sdata: [31] [17] ...

sdata[0] = 31  ✓
```

With 256 threads this takes only **8 steps** (log2(256)) instead of 256 steps.
Each step halves the active threads — like a tournament bracket.

**Why `__syncthreads()` is needed:**
After each step, every thread must have finished writing before the next
step reads. Without the barrier, fast threads would read stale values
written by slow threads from the previous step.

---

## Approach 3 — Warp Shuffle

A warp is a group of 32 threads that execute together on the same hardware unit.
They share a register file, so they can read each other's registers directly —
no shared memory, no barrier needed.

`__shfl_down_sync(mask, val, offset)` — thread `tid` receives the value
from thread `tid + offset` in the same warp.

```
Warp of 8 lanes (simplified), values: [3, 1, 4, 1, 5, 9, 2, 6]

offset=4:  lane 0 += lane 4   lane 1 += lane 5   lane 2 += lane 6   lane 3 += lane 7
           [8,  10, 6,  7,  5, 9, 2, 6]

offset=2:  lane 0 += lane 2   lane 1 += lane 3
           [14, 17, 6,  7,  ...]

offset=1:  lane 0 += lane 1
           [31, ...]

lane 0 = 31  ✓
```

A full 32-lane warp reduces in **5 instructions** (log2(32)).
No shared memory. No synchronisation barriers. Hardware-level speed.

---

## Comparison

| Approach | Steps | Memory used | Sync needed |
|---|---|---|---|
| Naive | N | none | none |
| Shared tree | log2(BLOCK) | shared memory | `__syncthreads()` after each step |
| Warp shuffle | log2(32) = 5 | registers only | none |

Expected timings on 16M floats (RTX 2000 Ada):

```
Naive  : ~100 ms   (one thread doing 16M additions)
Shared :  ~0.5 ms  (65536 blocks working in parallel)
Warp   :  ~0.3 ms  (same but faster per-warp primitive)
```

---

## Sum vs Max — Why You Need Both

The most important use of reduction in deep learning is **softmax**:

```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for all j)
```

Naively computing this has a problem. If any x[i] is large (e.g. 100),
exp(100) = 2.7e43 which overflows float32.

The fix — **numerically stable softmax** — uses two reductions:

```
Step 1 (max reduction):  m = max(x)
Step 2 (shift):          x_safe = x - m        # all values <= 0, no overflow
Step 3 (exp + sum):      s = sum(exp(x_safe))
Step 4 (divide):         output = exp(x_safe) / s
```

This gives identical mathematical results but never overflows.

**This is exactly why Flash Attention needs efficient max and sum reductions.**
Every attention score row goes through this two-reduction softmax,
and Flash Attention does it in a single GPU pass using warp shuffles
to compute the running max and sum as blocks of the matrix are loaded.

---

## Files in This Module

```
03_reduction/
    reduce.cu    Five kernels with timing: naive, shared, warp, warp_fixed, CUB
```

## Build & Run

```bash
cd 03_reduction
nvcc -O2 -arch=sm_89 -I.. reduce.cu -o reduce
./reduce
```

## What to Notice

- Naive is dramatically slower — one thread doing 16M additions serially
- Shared and warp are close; warp slightly faster (no shared memory overhead)
- `reduce_warp` has one `atomicAdd` per warp (8 per block); `reduce_warp_block`
  collapses those into one per block using shared memory + a second shuffle pass
- CUB is the fastest — it uses a multi-element-per-thread strategy plus
  prefetching that the hand-written kernels do not
