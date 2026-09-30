#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>

#ifdef LAYERNORM_USE_WELFORD
#include "layernorm_welford.cuh"
namespace layernorm_impl = layernorm_welford;
#else
#include "layernorm_cub.cuh"
namespace layernorm_impl = layernorm_cub;
#endif

namespace {

void layernorm_out(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const at::Tensor& output,
    double epsilon) {
    for (const auto& tensor : {input, gamma, beta, output}) {
        TORCH_CHECK(tensor.is_cuda(), "all tensors must be CUDA tensors");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, "only FP32 is supported");
        TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
        TORCH_CHECK(tensor.device() == input.device(), "tensor devices must match");
    }
    TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(1) > 0,
                "input must have nonempty shape [M, N]");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == input.size(1),
                "gamma must have shape [N]");
    TORCH_CHECK(beta.dim() == 1 && beta.numel() == input.size(1),
                "beta must have shape [N]");
    TORCH_CHECK(!output.is_alias_of(input) && !output.is_alias_of(gamma)
                    && !output.is_alias_of(beta),
                "output must not share storage with inputs");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0,
                "epsilon must be finite and positive");
    if (input.size(1) % 4 == 0) {
        for (const auto& tensor : {input, gamma, beta, output}) {
            TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0,
                        "the vec4 path requires 16-byte aligned pointers");
        }
    }

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    const cudaError_t error = layernorm_impl::launch(
        input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), input.size(0), input.size(1),
        static_cast<float>(epsilon), stream.stream());
    TORCH_CHECK(error == cudaSuccess, "layernorm launch failed: ",
                cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("out", &layernorm_out, "Run the LayerNorm kernel on the current CUDA stream");
}
