#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace layernorm_v1 {

struct LaunchConfig {
    int vector_size;
    int block_size;
};

LaunchConfig choose_launch_config(std::size_t M, std::size_t N);

cudaError_t launch(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon,
    cudaStream_t stream = nullptr);

}  // namespace layernorm_v1
