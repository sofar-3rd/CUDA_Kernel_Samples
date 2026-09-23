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

    const std::size_t stride = blockDim.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warp_num = blockDim.x / warpSize;

    // 一个 block 最多支持 32 个 warp.
    __shared__ float warp_sums[32];

    float value = 0.0f;

    // 每个线程计算当前行中自己负责元素的局部和.
    for (std::size_t col = tid; col < N; col += stride) {
        value += input[row * N + col];
    }

    // 在每个 warp 内归约局部和, 并将结果汇总到 lane 0.
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    // 由线程 0 汇总当前 block 内所有 warp 的结果.
    if (tid == 0) {
        float sum = 0.0f;
        for (int warp = 0; warp < warp_num; ++warp) {
            sum += warp_sums[warp];
        }
        output[row] = sum / static_cast<float>(N);
    }
}

void initialize_data(float* data, std::size_t n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

void compute_means_cpu(
    const float* input,
    float* means,
    std::size_t M,
    std::size_t N) {

    for (std::size_t row = 0; row < M; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < N; ++col) {
            sum += input[row * N + col];
        }
        means[row] = sum / static_cast<float>(N);
    }
}

bool verify_means(
    const float* actual,
    const float* expected,
    std::size_t M) {

    constexpr float absolute_tolerance = 1e-5f;
    constexpr float relative_tolerance = 1e-5f;

    for (std::size_t row = 0; row < M; ++row) {
        const float difference = std::fabs(actual[row] - expected[row]);
        const float tolerance = absolute_tolerance
                              + relative_tolerance * std::fabs(expected[row]);
        if (difference > tolerance) {
            std::fprintf(stderr,
                         "Mean mismatch at row %zu: GPU=%f, CPU=%f\n",
                         row, actual[row], expected[row]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr std::size_t M = 1024;
    constexpr std::size_t N = 2048;
    constexpr int THREADS = 128;

    const std::size_t input_bytes = sizeof(float) * M * N;
    const std::size_t means_bytes = sizeof(float) * M;

    float* h_input = static_cast<float*>(std::malloc(input_bytes));
    float* h_means = static_cast<float*>(std::malloc(means_bytes));
    float* h_means_reference = static_cast<float*>(std::malloc(means_bytes));

    if (h_input == nullptr || h_means == nullptr || h_means_reference == nullptr) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::free(h_input);
        std::free(h_means);
        std::free(h_means_reference);
        return EXIT_FAILURE;
    }

    initialize_data(h_input, M * N);
    compute_means_cpu(h_input, h_means_reference, M, N);

    float* d_input = nullptr;
    float* d_means = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_means), means_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    const dim3 block(THREADS);
    const dim3 grid(M);

    // gamma, beta 和 epsilon 会在后续完整 LayerNorm 实现中使用.
    layernorm_kernel<<<grid, block>>>(
        d_input,
        nullptr,
        nullptr,
        d_means,
        M,
        N,
        1e-5f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_means,
        d_means,
        means_bytes,
        cudaMemcpyDeviceToHost));

    const bool passed = verify_means(h_means, h_means_reference, M);
    std::printf("LayerNorm mean verification: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_means));
    std::free(h_input);
    std::free(h_means);
    std::free(h_means_reference);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
