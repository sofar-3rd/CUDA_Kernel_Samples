#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

namespace layernorm_test {

constexpr float kEpsilon = 1e-5f;

struct TestCase {
    const char* name;
    std::size_t M;
    std::size_t N;
    int expected_vector_size;
    int expected_block_size;
};

inline bool check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

inline void initialize_input(std::vector<float>& input) {
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (float& value : input) {
        value = distribution(generator);
    }
}

inline void initialize_affine(
    std::vector<float>& gamma,
    std::vector<float>& beta) {
    for (std::size_t col = 0; col < gamma.size(); ++col) {
        gamma[col] = 0.5f + static_cast<float>(col % 17) / 16.0f;
        beta[col] = -0.25f + static_cast<float>(col % 13) / 24.0f;
    }
}

inline void reference(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    std::vector<float>& output,
    std::size_t M,
    std::size_t N) {
    for (std::size_t row = 0; row < M; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < N; ++col) {
            sum += input[row * N + col];
        }
        const double mean = sum / static_cast<double>(N);

        double squared_difference_sum = 0.0;
        for (std::size_t col = 0; col < N; ++col) {
            const double difference = input[row * N + col] - mean;
            squared_difference_sum += difference * difference;
        }
        const double inverse_std = 1.0 / std::sqrt(
            squared_difference_sum / static_cast<double>(N) + kEpsilon);

        for (std::size_t col = 0; col < N; ++col) {
            const std::size_t index = row * N + col;
            const double normalized = (input[index] - mean) * inverse_std;
            output[index] = static_cast<float>(
                normalized * gamma[col] + beta[col]);
        }
    }
}

inline bool compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const TestCase& test_case) {
    constexpr float absolute_tolerance = 2e-4f;
    constexpr float relative_tolerance = 2e-4f;
    float max_absolute_error = 0.0f;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            std::fprintf(stderr, "%s produced non-finite output at %zu\n",
                         test_case.name, index);
            return false;
        }
        const float absolute_error = std::fabs(actual[index] - expected[index]);
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        const float tolerance = absolute_tolerance
            + relative_tolerance * std::fabs(expected[index]);
        if (absolute_error > tolerance) {
            std::fprintf(
                stderr,
                "%s mismatch at %zu: GPU=%f reference=%f error=%g tolerance=%g\n",
                test_case.name,
                index,
                actual[index],
                expected[index],
                absolute_error,
                tolerance);
            return false;
        }
    }

    std::printf("[PASS] %-12s M=%zu N=%zu max_abs=%g\n",
                test_case.name, test_case.M, test_case.N, max_absolute_error);
    return true;
}

template <typename ChooseConfig, typename Launch>
bool run_case(
    const TestCase& test_case,
    ChooseConfig choose_config,
    Launch launch) {
    const auto config = choose_config(test_case.M, test_case.N);
    if (config.vector_size != test_case.expected_vector_size
        || config.block_size != test_case.expected_block_size) {
        std::fprintf(
            stderr,
            "%s config mismatch: got vec=%d block=%d, expected vec=%d block=%d\n",
            test_case.name,
            config.vector_size,
            config.block_size,
            test_case.expected_vector_size,
            test_case.expected_block_size);
        return false;
    }

    const std::size_t element_count = test_case.M * test_case.N;
    const std::size_t tensor_bytes = element_count * sizeof(float);
    const std::size_t parameter_bytes = test_case.N * sizeof(float);
    std::vector<float> input(element_count);
    std::vector<float> gamma(test_case.N);
    std::vector<float> beta(test_case.N);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    initialize_input(input);
    initialize_affine(gamma, beta);
    reference(input, gamma, beta, expected, test_case.M, test_case.N);

    float* device_input = nullptr;
    float* device_gamma = nullptr;
    float* device_beta = nullptr;
    float* device_output = nullptr;
    bool ok = check_cuda(cudaMalloc(&device_input, tensor_bytes), "cudaMalloc input")
        && check_cuda(cudaMalloc(&device_gamma, parameter_bytes), "cudaMalloc gamma")
        && check_cuda(cudaMalloc(&device_beta, parameter_bytes), "cudaMalloc beta")
        && check_cuda(cudaMalloc(&device_output, tensor_bytes), "cudaMalloc output")
        && check_cuda(cudaMemcpy(device_input, input.data(), tensor_bytes,
                                 cudaMemcpyHostToDevice), "copy input")
        && check_cuda(cudaMemcpy(device_gamma, gamma.data(), parameter_bytes,
                                 cudaMemcpyHostToDevice), "copy gamma")
        && check_cuda(cudaMemcpy(device_beta, beta.data(), parameter_bytes,
                                 cudaMemcpyHostToDevice), "copy beta");

    if (ok) {
        ok = check_cuda(
            launch(device_input, device_gamma, device_beta, device_output,
                   test_case.M, test_case.N, kEpsilon, nullptr),
            "kernel launch");
    }
    if (ok) {
        ok = check_cuda(cudaStreamSynchronize(nullptr), "kernel execution");
    }
    if (ok) {
        ok = check_cuda(cudaMemcpy(actual.data(), device_output, tensor_bytes,
                                   cudaMemcpyDeviceToHost), "copy output");
    }
    if (ok) {
        ok = compare(actual, expected, test_case);
    }

    cudaFree(device_input);
    cudaFree(device_gamma);
    cudaFree(device_beta);
    cudaFree(device_output);
    return ok;
}

template <typename ChooseConfig, typename Launch>
int run_correctness_suite(
    const TestCase* test_cases,
    std::size_t test_case_count,
    ChooseConfig choose_config,
    Launch launch) {
    bool passed = true;
    for (std::size_t index = 0; index < test_case_count; ++index) {
        passed = run_case(test_cases[index], choose_config, launch) && passed;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

template <typename ChooseConfig, typename Launch>
int run_benchmark(
    const char* implementation_name,
    ChooseConfig choose_config,
    Launch launch) {
    constexpr std::size_t M = 1024;
    constexpr std::size_t N = 2048;
    constexpr int warmup_iterations = 100;
    constexpr int timed_iterations = 1000;
    const std::size_t tensor_bytes = M * N * sizeof(float);
    const std::size_t parameter_bytes = N * sizeof(float);
    std::vector<float> input(M * N);
    std::vector<float> gamma(N);
    std::vector<float> beta(N);
    initialize_input(input);
    initialize_affine(gamma, beta);

    float* device_input = nullptr;
    float* device_gamma = nullptr;
    float* device_beta = nullptr;
    float* device_output = nullptr;
    if (!check_cuda(cudaMalloc(&device_input, tensor_bytes), "cudaMalloc input")
        || !check_cuda(cudaMalloc(&device_gamma, parameter_bytes), "cudaMalloc gamma")
        || !check_cuda(cudaMalloc(&device_beta, parameter_bytes), "cudaMalloc beta")
        || !check_cuda(cudaMalloc(&device_output, tensor_bytes), "cudaMalloc output")
        || !check_cuda(cudaMemcpy(device_input, input.data(), tensor_bytes,
                                  cudaMemcpyHostToDevice), "copy input")
        || !check_cuda(cudaMemcpy(device_gamma, gamma.data(), parameter_bytes,
                                  cudaMemcpyHostToDevice), "copy gamma")
        || !check_cuda(cudaMemcpy(device_beta, beta.data(), parameter_bytes,
                                  cudaMemcpyHostToDevice), "copy beta")) {
        return EXIT_FAILURE;
    }

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (!check_cuda(
                launch(device_input, device_gamma, device_beta, device_output,
                       M, N, kEpsilon, nullptr), "warmup launch")) {
            return EXIT_FAILURE;
        }
    }
    if (!check_cuda(cudaStreamSynchronize(nullptr), "warmup execution")) {
        return EXIT_FAILURE;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start event");
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        if (!check_cuda(
                launch(device_input, device_gamma, device_beta, device_output,
                       M, N, kEpsilon, nullptr), "timed launch")) {
            return EXIT_FAILURE;
        }
    }
    check_cuda(cudaEventRecord(stop), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for stop event");

    float total_milliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_milliseconds, start, stop),
               "measure elapsed time");
    const double average_microseconds =
        total_milliseconds * 1000.0 / timed_iterations;
    const double algorithm_bytes = static_cast<double>(8 * M * N + 8 * N);
    const double effective_gigabytes_per_second =
        algorithm_bytes / (average_microseconds * 1e3);
    const auto config = choose_config(M, N);
    std::printf(
        "[BENCH] %s M=%zu N=%zu vec=%d block=%d time=%.3f us effective=%.2f GB/s\n",
        implementation_name,
        M,
        N,
        config.vector_size,
        config.block_size,
        average_microseconds,
        effective_gigabytes_per_second);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_input);
    cudaFree(device_gamma);
    cudaFree(device_beta);
    cudaFree(device_output);
    return EXIT_SUCCESS;
}

template <typename ChooseConfig, typename Launch>
int run_main(
    int argc,
    char** argv,
    const char* implementation_name,
    const TestCase* test_cases,
    std::size_t test_case_count,
    ChooseConfig choose_config,
    Launch launch) {
    if (argc == 1) {
        return run_correctness_suite(
            test_cases, test_case_count, choose_config, launch);
    }
    if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
        return run_benchmark(implementation_name, choose_config, launch);
    }
    std::fprintf(stderr, "Usage: %s [--benchmark]\n", argv[0]);
    return EXIT_FAILURE;
}

}  // namespace layernorm_test
