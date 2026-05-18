/*
 * pybind11 binding file.
 *
 * This file is the bridge between Python and your CUDA code.
 * It declares what functions Python can call and maps them to C++ symbols.
 *
 * TORCH_EXTENSION_NAME is a macro set by the build system to the module name
 * you choose in setup.py (here: "vec_add_ext").
 */

#include <torch/extension.h>

// Forward declaration -- the actual definition is in vec_add_kernel.cu
torch::Tensor vec_add_cuda(torch::Tensor A, torch::Tensor B);

// This macro defines the Python module and its exported functions.
// m.def("python_name", &cpp_function, "docstring")
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add_cuda, "Vector addition CUDA kernel");
}
