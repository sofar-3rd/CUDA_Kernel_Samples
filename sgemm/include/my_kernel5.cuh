#pragma once
#include <cuda_runtime.h>
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/*
固定住 C 的一个方块
A 的窗口不断向右滑
B 的窗口不断向下滑
每滑动一次，产生一个部分乘积
所有部分乘积累加到 local_sum
*/

// 引入二维 thread tile，每个线程负责 C 的连续 TM 行 TN 列

template <const int BM,
          const int BN,
          const int BK,
          const int TM,
          const int TN>
__global__ void my_sgemm_v5(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int thread_c_col = threadIdx.x % (BN / TN);
    const int thread_c_row = threadIdx.x / (BN / TN);

    const int thread_a_col = threadIdx.x % (BK / 4);
    const int thread_a_row = threadIdx.x / (BK / 4);
    const int thread_a_stride = blockDim.x * 4 / BK;  // A tile 的行数为 BK, 所有线程一次能搬运 blockDim.x * 4, 记录一次搬运几行
    const int thread_b_col = threadIdx.x % (BN / 4);
    const int thread_b_row = threadIdx.x / (BN / 4);
    const int thread_b_stride = blockDim.x * 4 / BN;  // B tile 的行数为 BN, 所有线程一次能搬运 blockDim.x * 4, 记录一次搬运几行


    // 移动 A/B/C 指针到指定的 tile 处
    A = A + by * BM * K;
    B = B + bx * BN;
    C = C + by * BM * N + bx * BN;

    __shared__ float As[BK][BM]; // 转置
    __shared__ float Bs[BK][BN];

    float res_reg[TM][TN] = {0.f}; // 一个线程处理 TM * TN 个计算结果
    float A_reg[TM];
    float B_reg[TN];

    // 外层 block 循环
    for(int i=0 ; i<K; i+=BK){
        // 搬运 A tile 到共享内存
        for(int s=0; s<BM; s+=thread_a_stride){
            float4 tmp = FLOAT4(A[(thread_a_row + s) * K + thread_a_col * 4]);
            As[thread_a_col * 4][thread_a_row + s]     = tmp.x;
            As[thread_a_col * 4 + 1][thread_a_row + s] = tmp.y;
            As[thread_a_col * 4 + 2][thread_a_row + s] = tmp.z;
            As[thread_a_col * 4 + 3][thread_a_row + s] = tmp.w;
        }
        for(int s=0; s<BK; s+=thread_b_stride){
            // 搬运 B tile 到共享内存
            FLOAT4(Bs[thread_b_row + s][thread_b_col * 4]) = FLOAT4(B[(thread_b_row + s) * N + (thread_b_col * 4)]);
        }
        __syncthreads();

        // 移动 A/B 指针到指定的 block 处
        A += BK;
        B += BK * N;

        // 一个线程处理 TM 行 TN 列的 C 元素
#pragma unroll
        for (int dotIdx=0; dotIdx<BK; ++dotIdx){

#pragma unroll
            for (int resIdxM=0; resIdxM<TM; resIdxM+=4){
                FLOAT4(A_reg[resIdxM]) = FLOAT4(As[dotIdx][thread_c_row * TM + resIdxM]);
            }

#pragma unroll
            for (int resIdxN=0; resIdxN<TN; resIdxN+=4){
                FLOAT4(B_reg[resIdxN]) = FLOAT4(Bs[dotIdx][thread_c_col * TM + resIdxN]);
            }

#pragma unroll
            for (int resIdxM=0; resIdxM<TM; ++resIdxM){
#pragma unroll
                for (int resIdxN=0; resIdxN<TN; ++resIdxN){
                    res_reg[resIdxM][resIdxN] += A_reg[resIdxM] * B_reg[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    // 将结果写入 C 矩阵中
#pragma unroll
    for (int resIdxM=0; resIdxM<TM; ++resIdxM){
#pragma unroll
        for (int resIdxN=0; resIdxN<TN; resIdxN+=4){
            const int offset = (thread_c_row * TM + resIdxM) * N + (thread_c_col * TN + resIdxN);
            float4 tmp = FLOAT4(C[offset]);
            tmp.x = alpha * res_reg[resIdxM][resIdxN] + beta * tmp.x;
            tmp.y = alpha * res_reg[resIdxM][resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * res_reg[resIdxM][resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * res_reg[resIdxM][resIdxN + 3] + beta * tmp.w;
            FLOAT4(C[offset]) = tmp;
        }
    }
}