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

// 仅支持 M=N=K、正边长且边长为 BLOCK_SIZE 整数倍的方阵。
template<const int BLOCK_SIZE>
inline bool my_sgemm_v2_supported(int M, int N, int K) {
    return M > 0 && M == N && N == K && M % BLOCK_SIZE == 0;
}

template<const int BLOCK_SIZE>
__global__ void my_sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    constexpr int BK = BLOCK_SIZE;

    int bx = blockIdx.x; // block 列
    int by = blockIdx.y; // block 行

    int tx = threadIdx.x % BN; // block 中 element 列
    int ty = threadIdx.x / BN; // block 中 element 行

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float* p_A_start = &A[by * BM * K];             // 矩阵 A_tile 起始点
    float* p_B_start = &B[bx * BN];                 // 矩阵 B_tile 起始点
    float* p_C_start = &C[by * BM * N + bx * BN];   // 矩阵 C_tile 起始点

    float local_sum = 0.f;
    for(int i=0; i<K; i+=BK){
        As[ty][tx] = p_A_start[ty * K + (tx + i)]; // 缓存 A_tile
        Bs[ty][tx] = p_B_start[(ty + i) * N + tx]; // 缓存 B_tile
        __syncthreads();

        for(int j=0; j<BK; ++j){
            local_sum += As[ty][j] * Bs[j][tx];
        }
        __syncthreads();
    }
    int offset = ty * N + tx;
    p_C_start[offset] = alpha * local_sum + beta * p_C_start[offset];
}