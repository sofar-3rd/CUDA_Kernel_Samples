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

int main() {
    constexpr unsigned int m = 512, n = 512, k = 512;
    constexpr unsigned int BM = 32, BN = 32, BK = 16;
    constexpr float alpha = 1.25f, beta = 0.5f;

    float a[m * k] = {}, b[k * n] = {}, c[m * n] = {}, result[m * n] = {};
    std::vector<float> expected(m * n);
    randomize_matrix(a, m * k);
    randomize_matrix(b, k * n);
    randomize_matrix(c, m * n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    cudaCheck(cudaMalloc(&d_a, sizeof(a)));
    cudaCheck(cudaMalloc(&d_b, sizeof(b)));
    cudaCheck(cudaMalloc(&d_c, sizeof(c)));
    cudaCheck(cudaMalloc(&d_ref, sizeof(c)));
    cudaCheck(cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_c, c, sizeof(c), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_ref, c, sizeof(c), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));
    // 列在前，行在后，blocksize = thread
    my_sgemm_v2<BM, BN, BK><<<dim3(n / BN, m / BM), max(BM * BN, max(BM * BK, BN * BK))>>>(
        m, n, k, alpha, d_a, d_b, beta, d_c);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_ms = 0.f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaMemcpy(result, d_c, sizeof(result), cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle));
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, d_b, n, d_a, k, &beta, d_ref, n));
    cudaCheck(cudaMemcpy(expected.data(), d_ref, sizeof(c), cudaMemcpyDeviceToHost));
    check_cublas(cublasDestroy(handle));

    bool passed = true;
    for (unsigned int row = 0; row < m; ++row) {
        for (unsigned int col = 0; col < n; ++col) {
            const float expected_value = expected[row * n + col];
            const float actual = result[row * n + col];
            if (!std::isfinite(actual) || !std::isfinite(expected_value) ||
                std::fabs(actual - expected_value) > 1e-4f + 1e-4f * std::fabs(expected_value)) {
                std::fprintf(stderr, "FAIL (%u,%u): expected=%g actual=%g\n",
                             row, col, expected_value, actual);
                passed = false;
            }
        }
    }

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    cudaCheck(cudaFree(d_ref));
    std::printf("Kernel time: %.3f ms\n", elapsed_ms);
    if (passed) std::puts("PASS my_kernel2_BM_BN");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}