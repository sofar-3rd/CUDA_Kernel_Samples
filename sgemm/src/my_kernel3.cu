#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <my_kernel3.cuh>

static void check_cuda(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define cudaCheck(expr) check_cuda((expr), __FILE__, __LINE__)

static void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

template <int BM, int BN, int BK, int TM>
static bool run_case(unsigned int m, unsigned int n, unsigned int k) {
    constexpr float alpha = 1.25f, beta = 0.5f;
    std::vector<float> a(m * k), b(k * n), c(m * n), result(m * n), expected(m * n);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<int>((i * 17) % 101) - 50) * 0.01f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = (static_cast<int>((i * 29) % 103) - 51) * 0.01f;
    for (size_t i = 0; i < c.size(); ++i) c[i] = (static_cast<int>((i * 11) % 107) - 53) * 0.01f;

    const size_t a_bytes = a.size() * sizeof(float);
    const size_t b_bytes = b.size() * sizeof(float);
    const size_t c_bytes = c.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    cudaCheck(cudaMalloc(&d_a, a_bytes));
    cudaCheck(cudaMalloc(&d_b, b_bytes));
    cudaCheck(cudaMalloc(&d_c, c_bytes));
    cudaCheck(cudaMalloc(&d_ref, c_bytes));
    cudaCheck(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_c, c.data(), c_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_ref, c.data(), c_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));
    my_sgemm_v3<BM, BN, BK, TM><<<dim3((n + BN - 1) / BN, (m + BM - 1) / BM), BM * BN / TM>>>(
        m, n, k, alpha, d_a, d_b, beta, d_c);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_ms = 0.f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaMemcpy(result.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle));
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, d_b, n, d_a, k, &beta, d_ref, n));
    cudaCheck(cudaMemcpy(expected.data(), d_ref, c_bytes, cudaMemcpyDeviceToHost));
    check_cublas(cublasDestroy(handle));

    bool passed = true;
    for (unsigned int row = 0; row < m; ++row) {
        for (unsigned int col = 0; col < n; ++col) {
            const float expected_value = expected[row * n + col];
            const float actual = result[row * n + col];
            if (!std::isfinite(actual) || !std::isfinite(expected_value) ||
                std::fabs(actual - expected_value) > 1e-4f + 1e-4f * std::fabs(expected_value)) {
                if (passed) std::fprintf(stderr, "FAIL M=%u N=%u K=%u BM=%d BN=%d BK=%d TM=%d\n",
                                         m, n, k, BM, BN, BK, TM);
                if (passed) std::fprintf(stderr, "  (%u,%u): expected=%g actual=%g\n",
                                         row, col, expected_value, actual);
                passed = false;
            }
        }
    }

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    cudaCheck(cudaFree(d_ref));
    if (passed) std::printf("PASS M=%u N=%u K=%u BM=%d BN=%d BK=%d TM=%d (%.3f ms)\n",
                          m, n, k, BM, BN, BK, TM, elapsed_ms);
    return passed;
}

int main() {
    bool passed = true;
    passed = run_case<64, 64, 8, 8>(64, 64, 64) && passed;
    passed = run_case<64, 64, 8, 8>(128, 128, 128) && passed;
    passed = run_case<64, 64, 8, 8>(512, 512, 512) && passed;
    passed = run_case<64, 64, 8, 8>(768, 768, 768) && passed;
    passed = run_case<64, 64, 8, 8>(1024, 1024, 1024) && passed;
    passed = run_case<64, 64, 8, 8>(1280, 1280, 1280) && passed;
    passed = run_case<64, 64, 8, 8>(2048, 2048, 2048) && passed;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
