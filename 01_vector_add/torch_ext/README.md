# vec_add PyTorch Extension

Calls a custom CUDA kernel from Python using a PyTorch C++ extension.

---

## Prerequisites (on the VM)

- CUDA Toolkit installed (`nvcc --version` should work)
- PyTorch installed with CUDA support
- Python 3.8+

Verify PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# should print: True
```

---

## Step 1 — Find your GPU's compute capability

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Example output:
```
name, compute_cap
Tesla T4, 7.5        <-- sm_75
A10,      8.6        <-- sm_86
A100,     8.0        <-- sm_80
```

---

## Step 2 — Update the arch flag in setup.py

Open `setup.py` and change `sm_80` to match your GPU:

```python
"nvcc": ["-O2", "-arch=sm_75"],   # example for T4
```

---

## Step 3 — Build the extension

```bash
cd 01_vector_add/torch_ext
python setup.py build_ext --inplace
```

This compiles the `.cu` and `.cpp` files and produces a `.so` file in the
same directory, e.g. `vec_add_ext.cpython-310-x86_64-linux-gnu.so`.

You only need to rebuild when you change the `.cu` or `.cpp` files.

---

## Step 4 — Run

```bash
python run.py
```

Expected output:
```
Max error: 0.0
```

---

## What each file does

| File | Role |
|---|---|
| `vec_add_kernel.cu` | CUDA kernel + C++ launcher |
| `vec_add_ext.cpp` | pybind11 bindings (Python <-> C++ bridge) |
| `setup.py` | Build script, compiles everything into a `.so` |
| `run.py` | Python test script |

---

## Troubleshooting

**`nvcc` not found**
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**`torch/extension.h` not found**
PyTorch is not installed or was installed without CUDA. Reinstall:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**`CUDA error: no kernel image is available for execution`**
The `sm_XX` arch flag in `setup.py` does not match your GPU.
Repeat Step 1 and 2, then rebuild.

**Stale build after editing kernel**
```bash
rm -rf build/ *.so
python setup.py build_ext --inplace
```
