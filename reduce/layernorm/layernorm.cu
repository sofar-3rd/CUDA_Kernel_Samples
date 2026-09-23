#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "include/utils.cuh"

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ __forceinline__ void calculate_mean(
    const float* row_input,
    std::size_t N,
    float* warp_sums,
    float* shared_mean) {

    const int stride = blockDim.x;
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

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
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
    const int stride = blockDim.x;
    const std::size_t row_offset = row * N;

    if (row >= M) {
        return;
    }

    // 一个 block 最多支持 32 个 warp.
    __shared__ float warp_sums[32];
    __shared__ float shared_mean;
    __shared__ float shared_variance;

    // 计算均值
    calculate_mean(
        input + row_offset,
        N,
        warp_sums,
        &shared_mean);

    // 计算方差
    calculate_variance(
        input + row_offset,
        N,
        warp_sums,
        &shared_mean,
        &shared_variance);

    const float inverse_std = rsqrtf(shared_variance + epsilon);

    // 对行归一化
    for (std::size_t col = threadIdx.x; col < N; col += stride) {
        const std::size_t index = row_offset + col;
        const float normalized = (input[index] - shared_mean) * inverse_std;
        output[index] = normalized * gamma[col] + beta[col];
    }
}

void initialize_data(float* data, std::size_t n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

void compute_layernorm_output_cpu(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon) {

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
            const std::size_t index = row * N + col;
            const float normalized = (input[index] - mean)
                                   / std::sqrt(variance + epsilon);
            output[index] = normalized * gamma[col] + beta[col];
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
                         "LayerNorm mismatch at index %zu: GPU=%f, CPU=%f\n",
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
    constexpr float EPSILON = 1e-5f;

    const std::size_t element_count = M * N;
    const std::size_t tensor_bytes = sizeof(float) * element_count;
    const std::size_t parameter_bytes = sizeof(float) * N;

    float* h_input = static_cast<float*>(std::malloc(tensor_bytes));
    float* h_gamma = static_cast<float*>(std::malloc(parameter_bytes));
    float* h_beta = static_cast<float*>(std::malloc(parameter_bytes));
    float* h_output = static_cast<float*>(std::malloc(tensor_bytes));
    float* h_output_reference = static_cast<float*>(std::malloc(tensor_bytes));

    if (h_input == nullptr || h_gamma == nullptr || h_beta == nullptr
        || h_output == nullptr || h_output_reference == nullptr) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::free(h_input);
        std::free(h_gamma);
        std::free(h_beta);
        std::free(h_output);
        std::free(h_output_reference);
        return EXIT_FAILURE;
    }

    initialize_data(h_input, element_count);
    for (std::size_t col = 0; col < N; ++col) {
        h_gamma[col] = 0.5f + static_cast<float>(col % 17) / 16.0f;
        h_beta[col] = -0.25f + static_cast<float>(col % 13) / 24.0f;
    }
    compute_layernorm_output_cpu(
        h_input,
        h_gamma,
        h_beta,
        h_output_reference,
        M,
        N,
        EPSILON);

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_input), tensor_bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_gamma), parameter_bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_beta), parameter_bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_output), tensor_bytes));
    cudaCheck(cudaMemcpy(d_input, h_input, tensor_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_gamma, h_gamma, parameter_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_beta, h_beta, parameter_bytes, cudaMemcpyHostToDevice));

    const dim3 block(THREADS);
    const dim3 grid(M);

    layernorm_kernel<<<grid, block>>>(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        M,
        N,
        EPSILON);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(
        h_output,
        d_output,
        tensor_bytes,
        cudaMemcpyDeviceToHost));

    const bool passed = verify_output(
        h_output,
        h_output_reference,
        element_count);
    std::printf(
        "LayerNorm verification: %s\n",
        passed ? "PASS" : "FAIL");

    if (passed) {
        constexpr int CPU_REPEAT = 10;
        constexpr int GPU_REPEAT = 1000;

        volatile float cpu_checksum = 0.0f;
        const auto cpu_start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < CPU_REPEAT; ++repeat) {
            compute_layernorm_output_cpu(
                h_input,
                h_gamma,
                h_beta,
                h_output_reference,
                M,
                N,
                EPSILON);
            cpu_checksum += h_output_reference[repeat % element_count];
        }
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_average_ms =
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count()
            / CPU_REPEAT;

        const float gpu_total_ms = TIME_RECORD(
            GPU_REPEAT,
            ([&] {
                layernorm_kernel<<<grid, block>>>(
                    d_input,
                    d_gamma,
                    d_beta,
                    d_output,
                    M,
                    N,
                    EPSILON);
            }));
        cudaCheck(cudaGetLastError());
        const float gpu_average_ms = gpu_total_ms / GPU_REPEAT;

        std::printf("CPU average time: %.6f ms\n", cpu_average_ms);
        std::printf("CUDA average time: %.6f ms\n", gpu_average_ms);
        std::printf("Speedup: %.2fx\n", cpu_average_ms / gpu_average_ms);
        (void)cpu_checksum;
    }

    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_gamma));
    cudaCheck(cudaFree(d_beta));
    cudaCheck(cudaFree(d_output));
    std::free(h_input);
    std::free(h_gamma);
    std::free(h_beta);
    std::free(h_output);
    std::free(h_output_reference);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
