#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(error));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ __forceinline__ void calculate_mean(
    const float* row_input,
    std::size_t N,
    float* warp_sums,
    float* shared_mean) {

    const std::size_t stride = blockDim.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warp_num = blockDim.x / warpSize;

    float local_sum = 0.0f;

    // 每个线程计算当前行中自己负责元素的局部和.
    for (std::size_t col = tid; col < N; col += stride) {
        local_sum += row_input[col];
    }

    // 在每个 warp 内归约局部和, 并将结果汇总到 lane 0.
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // 由线程 0 汇总当前 block 内所有 warp 的结果.
    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_num; ++warp) {
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
    float* shared_mean,
    float* shared_variance) {

    const std::size_t stride = blockDim.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warp_num = blockDim.x / warpSize;

    float local_squared_difference_sum = 0.0f;

    // 每个线程计算当前行中自己负责元素的局部平方差之和.
    for (std::size_t col = tid; col < N; col += stride) {
        local_squared_difference_sum += sq(row_input[col] - *shared_mean);
    }

    // 在每个 warp 内归约局部平方差之和, 并将结果汇总到 lane 0.
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_squared_difference_sum += __shfl_down_sync(
            0xffffffff,
            local_squared_difference_sum,
            offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = local_squared_difference_sum;
    }
    __syncthreads();

    // 由线程 0 汇总当前 block 内所有 warp 的结果.
    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_num; ++warp) {
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

    // 一个 block 最多支持 32 个 warp.
    __shared__ float warp_sums[32];
    __shared__ float shared_mean;
    __shared__ float shared_variance;

    // 计算均值
    calculate_mean(
        input + row * N,
        N,
        warp_sums,
        &shared_mean);

    // 计算方差
    calculate_variance(
        input + row * N,
        N,
        warp_sums,
        &shared_mean,
        &shared_variance);

    // 当前阶段将每行方差广播到整行, 用于验证最终输出布局.
    for (std::size_t col = threadIdx.x; col < N; col += blockDim.x) {
        output[row * N + col] = shared_variance;
    }
}

void initialize_data(float* data, std::size_t n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

void compute_variance_output_cpu(
    const float* input,
    float* output,
    std::size_t M,
    std::size_t N) {

    for (std::size_t row = 0; row < M; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < N; ++col) {
            sum += input[row * N + col];
        }
        const float mean = sum / static_cast<float>(N);

        float squared_difference_sum = 0.0f;
        for (std::size_t col = 0; col < N; ++col) {
            const float difference = input[row * N + col] - mean;
            squared_difference_sum += difference * difference;
        }
        const float variance = squared_difference_sum / static_cast<float>(N);

        for (std::size_t col = 0; col < N; ++col) {
            output[row * N + col] = variance;
        }
    }
}

bool verify_output(
    const float* actual,
    const float* expected,
    std::size_t element_count) {

    constexpr float absolute_tolerance = 1e-5f;
    constexpr float relative_tolerance = 1e-5f;

    for (std::size_t index = 0; index < element_count; ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float tolerance = absolute_tolerance
                              + relative_tolerance * std::fabs(expected[index]);
        if (difference > tolerance) {
            std::fprintf(stderr,
                         "Variance mismatch at index %zu: GPU=%f, CPU=%f\n",
                         index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr std::size_t M = 1024;
    constexpr std::size_t N = 2048;
    constexpr int THREADS = 128;

    const std::size_t element_count = M * N;
    const std::size_t tensor_bytes = sizeof(float) * element_count;

    float* h_input = static_cast<float*>(std::malloc(tensor_bytes));
    float* h_output = static_cast<float*>(std::malloc(tensor_bytes));
    float* h_output_reference = static_cast<float*>(std::malloc(tensor_bytes));

    if (h_input == nullptr || h_output == nullptr || h_output_reference == nullptr) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::free(h_input);
        std::free(h_output);
        std::free(h_output_reference);
        return EXIT_FAILURE;
    }

    initialize_data(h_input, element_count);
    compute_variance_output_cpu(h_input, h_output_reference, M, N);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), tensor_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, tensor_bytes, cudaMemcpyHostToDevice));

    const dim3 block(THREADS);
    const dim3 grid(M);

    // gamma, beta 和 epsilon 会在后续完整 LayerNorm 实现中使用.
    layernorm_kernel<<<grid, block>>>(
        d_input,
        nullptr,
        nullptr,
        d_output,
        M,
        N,
        1e-5f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        tensor_bytes,
        cudaMemcpyDeviceToHost));

    const bool passed = verify_output(
        h_output,
        h_output_reference,
        element_count);
    std::printf(
        "LayerNorm variance verification: %s\n",
        passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_output);
    std::free(h_output_reference);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
