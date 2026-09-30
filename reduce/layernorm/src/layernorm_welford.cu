#include "layernorm_welford.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxBlockSize = 1024;

using CubBlockReduce = cub::BlockReduce<layernorm_welford::WelfordState, kMaxBlockSize>;
using WelfordState = layernorm_welford::WelfordState;
using WelfordOp = layernorm_welford::WelfordOp;

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
    const float4 value = *reinterpret_cast<const float4*>(address);
    return {{value.x, value.y, value.z, value.w}};
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
    *reinterpret_cast<float4*>(address) = make_float4(
        value.values[0],
        value.values[1],
        value.values[2],
        value.values[3]);
}

__device__ __forceinline__ WelfordState block_reduce_welford(
    WelfordState value,
    CubBlockReduce::TempStorage& reduce_storage,
    int valid_threads) {
    return CubBlockReduce(reduce_storage).Reduce(value, WelfordOp{}, valid_threads);
}

template <int VectorSize>
__device__ __forceinline__ WelfordState calculate_welford_state(
    const float* row_input,
    std::size_t vector_count,
    CubBlockReduce::TempStorage& reduce_storage) {
    WelfordState local_state;

    for (std::size_t vector_index = threadIdx.x; vector_index < vector_count; vector_index += blockDim.x) {
        const FloatPack<VectorSize> input = load_vector<VectorSize>(
            row_input + vector_index * VectorSize);
        for (int component = 0; component < VectorSize; ++component) {
            layernorm_welford::welford_update(local_state, input.values[component]);
        }
    }

    return block_reduce_welford(local_state, reduce_storage, static_cast<int>(blockDim.x));
}

template <int VectorSize>
__global__ void layernorm_kernel_welford(
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

    __shared__ CubBlockReduce::TempStorage reduce_storage;
    __shared__ float shared_mean;
    __shared__ float shared_inverse_std;

    WelfordState state = calculate_welford_state<VectorSize>(
        row_input, vector_count, reduce_storage);
    if (threadIdx.x == 0) {
        shared_mean = state.mean;
        shared_inverse_std = rsqrtf(state.m2/state.count + epsilon);
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
    const std::size_t max_block_size = M < 256 ? 1024 : 256;
    return static_cast<int>(min(vector_count, max_block_size));
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
    layernorm_kernel_welford<VectorSize>
        <<<static_cast<unsigned int>(M), block_size, 0, stream>>>(
            input, gamma, beta, output, M, N, epsilon);
    return cudaGetLastError();
}

}  // namespace

namespace layernorm_welford {

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
    if (config.vector_size == 4
        && ((reinterpret_cast<std::uintptr_t>(input)
             | reinterpret_cast<std::uintptr_t>(gamma)
             | reinterpret_cast<std::uintptr_t>(beta)
             | reinterpret_cast<std::uintptr_t>(output)) & 15) != 0) {
        return cudaErrorInvalidValue;
    }
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

}  // namespace layernorm_welford
