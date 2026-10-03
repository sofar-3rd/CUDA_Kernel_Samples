#pragma once
#include <stddef.h>
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

// requirement: BM, BN, BK > 0; M, N > 0; K >= 0; tile edges are zero-padded
// max(BM * BN, BM * BK, BK * BN) <= maxThreadsPerBlock
// sizeof(float) * (BM * BK + BK * BN) <= sharedMemPerBlock

template<unsigned int BM, 
         unsigned int BN,
         unsigned int BK>
__global__ void my_sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    int bx = blockIdx.x; // block 列
    int by = blockIdx.y; // block 行

    int c_idx = threadIdx.x % BN; // matrix C block 中 element 列
    int c_idy = threadIdx.x / BN; // matrix C block 中 element 行

    int a_idx = threadIdx.x % BK;
    int a_idy = threadIdx.x / BK;

    int b_idx = threadIdx.x % BN;
    int b_idy = threadIdx.x / BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];             // 矩阵 A_tile 起始点
    B = &B[bx * BN];                 // 矩阵 B_tile 起始点
    C = &C[by * BM * N + bx * BN];   // 矩阵 C_tile 起始点

    float local_sum = 0.f;
    for(int i=0; i<K; i+=BK){
        if (a_idy < BM && a_idx < BK) {
            if (by * BM + a_idy < M && a_idx < K - i)
                As[a_idy][a_idx] = A[a_idy * K + a_idx];
            else
                As[a_idy][a_idx] = 0.f;
        }
        if (b_idy < BK && b_idx < BN) {
            if (b_idy < K - i && bx * BN + b_idx < N)
                Bs[b_idy][b_idx] = B[b_idy * N + b_idx];
            else
                Bs[b_idy][b_idx] = 0.f;
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        if(c_idy < BM && c_idx < BN){
            for(int j=0; j<BK; ++j)
                local_sum += As[c_idy][j] * Bs[j][c_idx];
        }
        __syncthreads();
    }
    if(c_idy < BM && c_idx < BN && by * BM + c_idy < M && bx * BN + c_idx < N){
        int offset = c_idy * N + c_idx;
        C[offset] = alpha * local_sum + beta * C[offset];
    }
}