#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "my_kernel2.cuh"

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

static void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

template<int BLOCK_SIZE>
static bool run_case(int M, int N, int K, float alpha, float beta) {
    std::vector<float> A(M * K), B(K * N), C(M * N), result(M * N), expected(M * N);
    for (size_t i = 0; i < A.size(); ++i) A[i] = (static_cast<int>(i % 13) - 6) / 7.f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = (static_cast<int>(i % 11) - 5) / 6.f;
    for (size_t i = 0; i < C.size(); ++i) C[i] = (static_cast<int>(i % 7) - 3) / 5.f;

    float *d_A, *d_B, *d_C, *d_ref;
    check(cudaMalloc(&d_A, A.size() * sizeof(float)));
    check(cudaMalloc(&d_B, B.size() * sizeof(float)));
    check(cudaMalloc(&d_C, C.size() * sizeof(float)));
    check(cudaMalloc(&d_ref, C.size() * sizeof(float)));
    check(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_C, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_ref, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    my_sgemm_v2<BLOCK_SIZE><<<grid, BLOCK_SIZE * BLOCK_SIZE>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(result.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    check(cublasCreate(&handle));
    check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha, d_B, N, d_A, K, &beta, d_ref, N));
    check(cudaMemcpy(expected.data(), d_ref, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cublasDestroy(handle));

    bool passed = true;
    for (int row = 0; row < M && passed; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected_value = expected[row * N + col];
            float actual = result[row * N + col];
            if (!std::isfinite(actual) || !std::isfinite(expected_value) ||
                std::fabs(expected_value - actual) > 1e-3f + 1e-4f * std::fabs(expected_value)) {
                std::fprintf(stderr, "FAIL tile=%d M=%d N=%d K=%d alpha=%g beta=%g at (%d,%d): expected=%g actual=%g\n",
                             BLOCK_SIZE, M, N, K, alpha, beta, row, col, expected_value, actual);
                passed = false;
                break;
            }
        }
    }
    check(cudaFree(d_A));
    check(cudaFree(d_B));
    check(cudaFree(d_C));
    check(cudaFree(d_ref));
    if (passed) std::printf("PASS tile=%d M=%d N=%d K=%d alpha=%g beta=%g\n", BLOCK_SIZE, M, N, K, alpha, beta);
    return passed;
}

int main() {
    bool passed = true;
    passed &= my_sgemm_v2_supported<32>(32, 32, 32);
    passed &= !my_sgemm_v2_supported<32>(32, 64, 32);
    passed &= !my_sgemm_v2_supported<32>(33, 33, 33);
    passed &= run_case<32>(32, 32, 32, 1.f, 0.f);
    passed &= run_case<32>(64, 64, 64, 1.25f, 0.5f);
    passed &= run_case<32>(256, 256, 256, -0.75f, 1.f);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
