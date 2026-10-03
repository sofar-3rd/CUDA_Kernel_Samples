#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <my_kernel2_BM_BN.cuh>
#include <utils.cuh>

int main() {
    constexpr unsigned int m = 512, n = 512, k = 512;
    constexpr unsigned int BM = 32, BN = 32, BK = 32;
    constexpr float alpha = 1.25f, beta = 0.5f;

    float a[m * k] = {}, b[k * n] = {}, c[m * n] = {}, result[m * n] = {};
    randomize_matrix(a, m * k);
    randomize_matrix(b, k * n);
    randomize_matrix(c, m * n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaCheck(cudaMalloc(&d_a, sizeof(a)));
    cudaCheck(cudaMalloc(&d_b, sizeof(b)));
    cudaCheck(cudaMalloc(&d_c, sizeof(c)));
    cudaCheck(cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_c, c, sizeof(c), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));
    // 列在前，行在后，blocksize = thread
    my_sgemm_v2<BM, BN, BK><<<dim3(n / BN, m / BM), BM * BN>>>(
        m, n, k, alpha, d_a, d_b, beta, d_c);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_ms = 0.f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaMemcpy(result, d_c, sizeof(result), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (unsigned int row = 0; row < m; ++row) {
        for (unsigned int col = 0; col < n; ++col) {
            float sum = 0.f;
            for (unsigned int i = 0; i < k; ++i) {
                sum += a[row * k + i] * b[i * n + col];
            }
            const float expected = alpha * sum + beta * c[row * n + col];
            const float actual = result[row * n + col];
            if (!std::isfinite(actual) ||
                std::fabs(actual - expected) > 1e-4f + 1e-4f * std::fabs(expected)) {
                std::fprintf(stderr, "FAIL (%u,%u): expected=%g actual=%g\n",
                             row, col, expected, actual);
                passed = false;
            }
        }
    }

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    std::printf("Kernel time: %.3f ms\n", elapsed_ms);
    if (passed) std::puts("PASS my_kernel2_BM_BN");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}