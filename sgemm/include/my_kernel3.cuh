#pragma once
#include <cuda_runtime.h>

/*
固定住 C 的一个方块
A 的窗口不断向右滑
B 的窗口不断向下滑
每滑动一次，产生一个部分乘积
所有部分乘积累加到 local_sum
*/

// 引入thread tile，每个线程负责 C 的同一列上连续 TM 行

template <const int BM, 
          const int BN, 
          const int BK, 
          const int TM>
__global__ void my_sgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int thread_c_col = threadIdx.x % BN;
    const int thread_c_row = threadIdx.x / BN;

    const int thread_a_col = threadIdx.x % BK;
    const int thread_a_row = threadIdx.x / BK;
    const int stride_a = blockDim.x / BK; // blockDim.x = thread_num = BM * BN / TM

    const int thread_b_col = threadIdx.x % BN;
    const int thread_b_row = threadIdx.x / BN;
    const int stride_b = blockDim.x / BN; // blockDim.x = thread_num = BM * BN / TM

    // 移动 A/B/C 指针到指定的 tile 处
    A = A + by * BM * K; 
    B = B + bx * BN;
    C = C + by * BM * N + bx * BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float temp[TM] = {0.f}; // 一个线程处理 TM 个计算结果

    // 外层 block 循环
    for(int i=0 ; i<K; i+=BK){
        // 搬运 A/B tile 到共享内存
#pragma unroll
        for (int s = 0; s < BM; s+=stride_a)
        {
            As[thread_a_row + s][thread_a_col] = A[(thread_a_row + s) * K + thread_a_col];
        }
#pragma unroll
        for (int s = 0; s < BK; s+=stride_b)
        {
            Bs[thread_b_row + s][thread_b_col] = B[(thread_b_row + s) * N + thread_b_col];
        }
        __syncthreads();

        // 移动 A/B 指针到指定的 block 处
        A += BK;
        B += BK * N;

        // 一个线程处理 TM 个 C 元素 (按列排布)
#pragma unroll
        for (int k=0; k<BK; ++k){
            float b_element = Bs[k][thread_c_col];
#pragma unroll
            for (int j=0; j<TM; ++j){
                temp[j] += As[thread_c_row * TM + j][k] * b_element;
            }
        }
        __syncthreads();
    }
    // 将结果写入 C 矩阵中
#pragma unroll
    for (int j=0; j<TM; ++j){
        const int offset = (thread_c_row * TM + j) * N + thread_c_col;
        C[offset] = alpha * temp[j] + beta * C[offset];
    }
}