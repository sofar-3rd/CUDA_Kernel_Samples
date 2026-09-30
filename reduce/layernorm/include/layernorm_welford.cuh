#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace layernorm_welford {

struct LaunchConfig {
    int vector_size;
    int block_size;
};

struct WelfordState {
    float mean = 0.f;
    float m2   = 0.f;   // Sum of squared deviations; variance = m2 / count.
    float count = 0.f;
};

__device__ __forceinline__ void welford_update(WelfordState& s, float x) {
    s.count += 1.f;
    const float delta = x - s.mean;
    s.mean += delta * (1.f / s.count);
    s.m2   += delta * (x - s.mean);
}

__device__ __forceinline__ WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (a.count == 0.f) return b;
    if (b.count == 0.f) return a;
    WelfordState r;
    r.count = a.count + b.count;
    const float delta = b.mean - a.mean;
    const float weight_b = b.count * (1.f / r.count);
    r.mean = a.mean + delta * weight_b;
    r.m2 = a.m2 + b.m2 + delta * delta * a.count * weight_b;
    return r;
}

struct WelfordOp {
    __device__ __forceinline__ WelfordState operator()(WelfordState a, WelfordState b) const {
        return welford_merge(a, b);
    }
};

LaunchConfig choose_launch_config(std::size_t M, std::size_t N);

// Contiguous FP32 tensors; output must not alias the inputs.
// For N divisible by 4, all pointers must be 16-byte aligned.
cudaError_t launch(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon,
    cudaStream_t stream = nullptr);

}  // namespace layernorm_welford
