#pragma once
#include <stdio.h>

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/*
固定住 C 的一个方块
A 的窗口不断向右滑
B 的窗口不断向下滑
每滑动一次，产生一个部分乘积
所有部分乘积累加到 local_sum
*/

template<unsigned int BM, 
         unsigned int BN,
         unsigned int BK>
__global__ void my_sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    int bx = blockIdx.x; // block 列
    int by = blockIdx.y; // block 行

    int tx = threadIdx.x % BN; // matrix C block 中 element 列
    int ty = threadIdx.x / BN; // matrix C block 中 element 行

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float* p_A_start = &A[by * BM * K];             // 矩阵 A_tile 起始点
    float* p_B_start = &B[bx * BN];                 // 矩阵 B_tile 起始点
    float* p_C_start = &C[by * BM * N + bx * BN];   // 矩阵 C_tile 起始点

    float local_sum = 0.f;
    for(int i=0; i<K; i+=BK){
#pragma unroll
        for (unsigned int index = threadIdx.x; index < BM * BK; index += blockDim.x) {
            As[index / BK][index % BK] = p_A_start[(index / BK) * K + i + index % BK];
        }
#pragma unroll
        for (unsigned int index = threadIdx.x; index < BK * BN; index += blockDim.x) {
            Bs[index / BN][index % BN] = p_B_start[(i + index / BN) * N + index % BN];
        }
        __syncthreads();

        for(int j=0; j<BK; ++j){
            local_sum += As[ty][j] * Bs[j][tx];
        }
        __syncthreads();
    }
    int offset = ty * N + tx;
    p_C_start[offset] = alpha * local_sum + beta * p_C_start[offset];
}