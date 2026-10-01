#include <my_kernel1.cuh>
#include <cuda_runtime.h>

__global__ void my_sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float local_sum = 0.0f;

    for(int i=0; i<K; ++i){
        local_sum += A[idy * K + i] * B[i * N + idx];
    }

    C[idy * N + idx] = local_sum * alpha + beta * C[idy * N + idx];
}