#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <my_kernel2_BM_BN.cuh>
#include <utils.cuh>

static void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

template <unsigned int BM, unsigned int BN, unsigned int BK>
static bool run_case(unsigned int m, unsigned int n, unsigned int k) {
    constexpr float alpha = 1.25f, beta = 0.5f;

    std::vector<float> a(m * k), b(k * n), c(m * n), result(m * n), expected(m * n);
    randomize_matrix(a.data(), a.size());
    randomize_matrix(b.data(), b.size());
    randomize_matrix(c.data(), c.size());

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    const size_t a_bytes = a.size() * sizeof(float);
    const size_t b_bytes = b.size() * sizeof(float);
    const size_t c_bytes = c.size() * sizeof(float);
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
    // 列在前，行在后，blocksize = thread
    my_sgemm_v2<BM, BN, BK><<<dim3(1 + (n - 1) / BN, 1 + (m - 1) / BM),
                                std::max({BM * BN, BM * BK, BN * BK})>>>(
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
                if (passed) std::fprintf(stderr, "FAIL M=%u N=%u K=%u BM=%u BN=%u BK=%u (%u,%u): expected=%g actual=%g\n",
                                         m, n, k, BM, BN, BK, row, col, expected_value, actual);
                passed = false;
            }
        }
    }

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    cudaCheck(cudaFree(d_ref));
    if (passed) std::printf("PASS M=%u N=%u K=%u BM=%u BN=%u BK=%u (%.3f ms)\n",
                            m, n, k, BM, BN, BK, elapsed_ms);
    return passed;
}

int main() {
    bool passed = true;
    passed = run_case<32, 32, 16>(512, 512, 512) && passed;
    passed = run_case<32, 32, 16>(65, 64, 16) && passed;
    passed = run_case<32, 32, 16>(64, 67, 16) && passed;
    passed = run_case<32, 32, 16>(64, 64, 19) && passed;
    passed = run_case<32, 32, 16>(33, 35, 17) && passed;
    passed = run_case<32, 32, 16>(7, 11, 3) && passed;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}