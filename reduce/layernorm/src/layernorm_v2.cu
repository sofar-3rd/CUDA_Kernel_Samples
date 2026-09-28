#include "layernorm_v2.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxWarpsPerBlock = 32;

template <int VectorSize>
struct FloatPack {
    float values[VectorSize];
};

template <int VectorSize>
__device__ __forceinline__ FloatPack<VectorSize> load_vector(
    const float* address) {
    FloatPack<VectorSize> result{};
    #pragma unroll
    for (int index = 0; index < VectorSize; ++index) {
        result.values[index] = address[index];
    }
    return result;
}

template <>
__device__ __forceinline__ FloatPack<4> load_vector<4>(
    const float* address) {
    FloatPack<4> result{};

    // TODO(student): Replace these scalar loads with one aligned float4 load.
    // The caller guarantees that N is divisible by 4. cudaMalloc provides an
    // aligned base address, so every row start is also 16-byte aligned.
    #pragma unroll
    for (int index = 0; index < 4; ++index) {
        result.values[index] = address[index];
    }
    return result;
}

template <int VectorSize>
__device__ __forceinline__ void store_vector(
    float* address,
    const FloatPack<VectorSize>& value) {
    #pragma unroll
    for (int index = 0; index < VectorSize; ++index) {
        address[index] = value.values[index];
    }
}

template <>
__device__ __forceinline__ void store_vector<4>(
    float* address,
    const FloatPack<4>& value) {
    // TODO(student): Replace these scalar stores with one aligned float4 store.
    #pragma unroll
    for (int index = 0; index < 4; ++index) {
        address[index] = value.values[index];
    }
}

__device__ __forceinline__ float block_reduce_sum(
    float value,
    float* warp_sums,
    float* block_sum) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_count = blockDim.x / kWarpSize;

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_count; ++warp) {
            sum += warp_sums[warp];
        }
        *block_sum = sum;
    }
    __syncthreads();
    return *block_sum;
}

template <int VectorSize>
__device__ __forceinline__ float calculate_mean(
    const float* row_input,
    std::size_t vector_count,
    std::size_t element_count,
    float* warp_sums,
    float* block_sum) {
    float accumulators[VectorSize] = {};

    for (std::size_t vector_index = threadIdx.x;
         vector_index < vector_count;
         vector_index += blockDim.x) {
        const FloatPack<VectorSize> input = load_vector<VectorSize>(
            row_input + vector_index * VectorSize);
        #pragma unroll
        for (int component = 0; component < VectorSize; ++component) {
            accumulators[component] += input.values[component];
        }
    }

    float local_sum = 0.0f;
    #pragma unroll
    for (int component = 0; component < VectorSize; ++component) {
        local_sum += accumulators[component];
    }

    const float sum = block_reduce_sum(local_sum, warp_sums, block_sum);
    return sum / static_cast<float>(element_count);
}

template <int VectorSize>
__device__ __forceinline__ float calculate_variance(
    const float* row_input,
    std::size_t vector_count,
    std::size_t element_count,
    float mean,
    float* warp_sums,
    float* block_sum) {
    float accumulators[VectorSize] = {};

    for (std::size_t vector_index = threadIdx.x;
         vector_index < vector_count;
         vector_index += blockDim.x) {
        const FloatPack<VectorSize> input = load_vector<VectorSize>(
            row_input + vector_index * VectorSize);
        #pragma unroll
        for (int component = 0; component < VectorSize; ++component) {
            const float difference = input.values[component] - mean;
            accumulators[component] += difference * difference;
        }
    }

    float local_sum = 0.0f;
    #pragma unroll
    for (int component = 0; component < VectorSize; ++component) {
        local_sum += accumulators[component];
    }

    const float sum = block_reduce_sum(local_sum, warp_sums, block_sum);
    return sum / static_cast<float>(element_count);
}

template <int VectorSize>
__global__ void layernorm_kernel_v2(
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
    const std::size_t vector_count = N / VectorSize;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    __shared__ float warp_sums[kMaxWarpsPerBlock];
    __shared__ float block_sum;
    __shared__ float shared_mean;
    __shared__ float shared_inverse_std;

    const float mean = calculate_mean<VectorSize>(
        row_input, vector_count, N, warp_sums, &block_sum);
    if (threadIdx.x == 0) {
        shared_mean = mean;
    }
    __syncthreads();

    const float variance = calculate_variance<VectorSize>(
        row_input,
        vector_count,
        N,
        shared_mean,
        warp_sums,
        &block_sum);
    if (threadIdx.x == 0) {
        shared_inverse_std = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    for (std::size_t vector_index = threadIdx.x;
         vector_index < vector_count;
         vector_index += blockDim.x) {
        const std::size_t element_offset = vector_index * VectorSize;
        const FloatPack<VectorSize> input_value = load_vector<VectorSize>(
            row_input + element_offset);
        const FloatPack<VectorSize> gamma_value = load_vector<VectorSize>(
            gamma + element_offset);
        const FloatPack<VectorSize> beta_value = load_vector<VectorSize>(
            beta + element_offset);
        FloatPack<VectorSize> output_value{};

        #pragma unroll
        for (int component = 0; component < VectorSize; ++component) {
            const float normalized =
                (input_value.values[component] - shared_mean)
                * shared_inverse_std;
            output_value.values[component] =
                normalized * gamma_value.values[component]
                + beta_value.values[component];
        }
        store_vector<VectorSize>(row_output + element_offset, output_value);
    }
}

int choose_vector_size(std::size_t N) {
    if (N % 4 == 0) {
        return 4;
    }
    if (N % 2 == 0) {
        return 2;
    }
    return 1;
}

int choose_block_size(std::size_t M, std::size_t vector_count) {
    // TODO(student): Select a power-of-two block size in [32, max_block_size].
    // Use max_block_size = 1024 when M < 256, otherwise 256. The current test
    // contract expects the largest power of two not greater than vector_count,
    // capped by max_block_size.
    (void)M;
    (void)vector_count;
    return 128;
}

template <int VectorSize>
cudaError_t launch_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon,
    int block_size,
    cudaStream_t stream) {
    layernorm_kernel_v2<VectorSize>
        <<<static_cast<unsigned int>(M), block_size, 0, stream>>>(
            input, gamma, beta, output, M, N, epsilon);
    return cudaGetLastError();
}

}  // namespace

namespace layernorm_v2 {

LaunchConfig choose_launch_config(std::size_t M, std::size_t N) {
    const int vector_size = choose_vector_size(N);
    const std::size_t vector_count = N / vector_size;
    return {vector_size, choose_block_size(M, vector_count)};
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
    if (M == 0 || N == 0) {
        return cudaErrorInvalidValue;
    }

    const LaunchConfig config = choose_launch_config(M, N);
    switch (config.vector_size) {
        case 4:
            return launch_kernel<4>(
                input, gamma, beta, output, M, N, epsilon,
                config.block_size, stream);
        case 2:
            return launch_kernel<2>(
                input, gamma, beta, output, M, N, epsilon,
                config.block_size, stream);
        case 1:
            return launch_kernel<1>(
                input, gamma, beta, output, M, N, epsilon,
                config.block_size, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace layernorm_v2
