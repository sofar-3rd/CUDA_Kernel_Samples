#include "layernorm.cuh"

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float square(float value) {
    return value * value;
}

__device__ __forceinline__ void calculate_mean(
    const float* row_input,
    std::size_t N,
    float* warp_sums,
    float* shared_mean) {
    const int stride = blockDim.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warp_count = blockDim.x / warpSize;
    float local_sum = 0.0f;

    for (std::size_t col = tid; col < N; col += stride) {
        local_sum += row_input[col];
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_count; ++warp) {
            sum += warp_sums[warp];
        }
        *shared_mean = sum / static_cast<float>(N);
    }
    __syncthreads();
}

__device__ __forceinline__ void calculate_variance(
    const float* row_input,
    std::size_t N,
    float* warp_sums,
    const float* shared_mean,
    float* shared_variance) {
    const int stride = blockDim.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warp_count = blockDim.x / warpSize;
    float local_sum = 0.0f;

    for (std::size_t col = tid; col < N; col += stride) {
        local_sum += square(row_input[col] - *shared_mean);
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_count; ++warp) {
            sum += warp_sums[warp];
        }
        *shared_variance = sum / static_cast<float>(N);
    }
    __syncthreads();
}

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    std::size_t M,
    std::size_t N,
    float epsilon) {
    const std::size_t row = blockIdx.x;
    if (row >= M) {
        return;
    }

    const std::size_t row_offset = row * N;
    __shared__ float warp_sums[32];
    __shared__ float shared_mean;
    __shared__ float shared_variance;

    calculate_mean(input + row_offset, N, warp_sums, &shared_mean);
    calculate_variance(
        input + row_offset,
        N,
        warp_sums,
        &shared_mean,
        &shared_variance);

    const float inverse_std = rsqrtf(shared_variance + epsilon);
    for (std::size_t col = threadIdx.x; col < N; col += blockDim.x) {
        const std::size_t index = row_offset + col;
        const float normalized = (input[index] - shared_mean) * inverse_std;
        output[index] = normalized * gamma[col] + beta[col];
    }
}

}  // namespace

namespace layernorm_v1 {

LaunchConfig choose_launch_config(std::size_t, std::size_t) {
    return {1, 128};
}

cudaError_t launch(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon,
    cudaStream_t stream) {
    const LaunchConfig config = choose_launch_config(M, N);
    layernorm_kernel<<<static_cast<unsigned int>(M), config.block_size, 0, stream>>>(
        input, gamma, beta, output, M, N, epsilon);
    return cudaGetLastError();
}

}  // namespace layernorm_v1
