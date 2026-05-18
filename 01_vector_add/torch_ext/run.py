"""
Test and profile the compiled extension.

Run this after building with: python setup.py build_ext --inplace
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import vec_add_ext   # the compiled .so built by setup.py

N = 1 << 20   # 1M elements

A = torch.ones(N, dtype=torch.float32, device="cuda")
B = torch.full((N,), 2.0, dtype=torch.float32, device="cuda")

# --- Correctness check -------------------------------------------------------
C = vec_add_ext.vec_add(A, B)
print("Max error:", (C - 3.0).abs().max().item())   # should print 0.0

# --- Warmup (first launch pays JIT / driver init costs) ----------------------
for _ in range(3):
    vec_add_ext.vec_add(A, B)
torch.cuda.synchronize()

# --- Profile -----------------------------------------------------------------
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("my_vec_add"):
        C = vec_add_ext.vec_add(A, B)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Save a Chrome trace -- open with chrome://tracing
prof.export_chrome_trace("vec_add_trace.json")
print("Trace saved to vec_add_trace.json")
