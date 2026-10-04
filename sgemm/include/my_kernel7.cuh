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
__global__ void my_sgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int thread_c_col = threadIdx.x % (BN / TN);
    const int thread_c_row = threadIdx.x / (BN / TN);

    const int thread_a_col = threadIdx.x % (BK / 4);
    const int thread_a_row = threadIdx.x / (BK / 4);
    const int thread_a_stride = blockDim.x * 4 / BK;

    const int thread_b_col = threadIdx.x % (BN / 4);
    const int thread_b_row = threadIdx.x / (BN / 4);
    const int thread_b_stride = blockDim.x * 4 / BN;  // B tile 的行数为 BN, 所有线程一次能搬运 blockDim.x * 4, 记录一次搬运几行

    constexpr int THREAD_NUM = BM * BN / (TM * TN);
    constexpr int A_LOADS_PER_THREAD = BM * BK / (THREAD_NUM * 4); // 每个线程要搬运多少个 FLOAT 4
    constexpr int B_LOADS_PER_THREAD = BK * BN / (THREAD_NUM * 4);


    // 移动 A/B/C 指针到指定的 tile 处
    A = A + by * BM * K;
    B = B + bx * BN;
    C = C + by * BM * N + bx * BN;

    __shared__ float As[2][BK][BM]; // 转置
    __shared__ float Bs[2][BK][BN];

    float res_reg[TM][TN] = {0.f}; // 一个线程处理 TM * TN 个计算结果
    float A_reg[2][TM];
    float B_reg[2][TN];
    float4 A_global_reg[A_LOADS_PER_THREAD];
    float4 B_global_reg[B_LOADS_PER_THREAD];

    int smem_write = 0;

    // 第一次加载 Global Memory 到 Shared Memory.
#pragma unroll
    for(int load_idx=0; load_idx<A_LOADS_PER_THREAD; ++load_idx){
        const int s = load_idx * thread_a_stride;
        float4 tmp = FLOAT4(A[(thread_a_row + s) * K + thread_a_col * 4]);
        As[smem_write][thread_a_col * 4][thread_a_row + s] = tmp.x;
        As[smem_write][thread_a_col * 4 + 1][thread_a_row + s] = tmp.y;
        As[smem_write][thread_a_col * 4 + 2][thread_a_row + s] = tmp.z;
        As[smem_write][thread_a_col * 4 + 3][thread_a_row + s] = tmp.w;
    }
#pragma unroll
    for(int load_idx=0; load_idx<B_LOADS_PER_THREAD; ++load_idx){
        const int s = load_idx * thread_b_stride;
        FLOAT4(Bs[smem_write][thread_b_row + s][thread_b_col * 4]) =
            FLOAT4(B[(thread_b_row + s) * N + thread_b_col * 4]);
    }
    __syncthreads();

    // 移动 A/B 指针到指定的 block 处
    A += BK;
    B += BK * N;


    // block 循环
    for(int i=BK ; i<K; i+=BK){
        smem_write ^= 1;
        // 将 next tile 预取到寄存器, 随后计算 current tile.
#pragma unroll
        for(int load_idx=0; load_idx<A_LOADS_PER_THREAD; ++load_idx){
            const int s = load_idx * thread_a_stride;
            A_global_reg[load_idx] = FLOAT4(A[(thread_a_row + s) * K + thread_a_col * 4]);
        }
#pragma unroll
        for(int load_idx=0; load_idx<B_LOADS_PER_THREAD; ++load_idx){
            const int s = load_idx * thread_b_stride;
            B_global_reg[load_idx] = FLOAT4(B[(thread_b_row + s) * N + thread_b_col * 4]);
        }

        A += BK;
        B += BK * N;

        /* ping pong smem2reg start */
#pragma unroll
        // As共享内存 到 A_reg寄存器
        for (int resIdxM=0; resIdxM<TM; resIdxM+=4)
            FLOAT4(A_reg[0][resIdxM]) = FLOAT4(As[smem_write ^ 1][0][thread_c_row * TM + resIdxM]);
#pragma unroll
        // Bs共享内存 到 B_reg寄存器
        for (int resIdxN=0; resIdxN<TN; resIdxN+=4)
            FLOAT4(B_reg[0][resIdxN]) = FLOAT4(Bs[smem_write ^ 1][0][thread_c_col * TN + resIdxN]);

        // 一个线程处理 TM 行 TN 列的 C 元素
#pragma unroll
        for (int dotIdx=1; dotIdx<BK; ++dotIdx){

#pragma unroll
            // As共享内存 到 A_reg寄存器
            for (int resIdxM=0; resIdxM<TM; resIdxM+=4){
                FLOAT4(A_reg[dotIdx % 2][resIdxM]) = FLOAT4(As[smem_write ^ 1][dotIdx][thread_c_row * TM + resIdxM]);
            }
#pragma unroll
            // Bs共享内存 到 B_reg寄存器
            for (int resIdxN=0; resIdxN<TN; resIdxN+=4){
                FLOAT4(B_reg[dotIdx % 2][resIdxN]) = FLOAT4(Bs[smem_write ^ 1][dotIdx][thread_c_col * TN + resIdxN]);
            }
#pragma unroll
            // 计算结果
            for (int resIdxM=0; resIdxM<TM; ++resIdxM){
                for (int resIdxN=0; resIdxN<TN; ++resIdxN){
                    res_reg[resIdxM][resIdxN] += A_reg[(dotIdx - 1) % 2][resIdxM] * B_reg[(dotIdx - 1) % 2][resIdxN];
                }
            }
        }
#pragma unroll
        // 计算最后一个block
        for (int resIdxM=0; resIdxM<TM; ++resIdxM){
            for (int resIdxN=0; resIdxN<TN; ++resIdxN){
                res_reg[resIdxM][resIdxN] += A_reg[(BK - 1) % 2][resIdxM] * B_reg[(BK - 1) % 2][resIdxN];
            }
        }

        // 计算完成后才消费预取寄存器, 将 next tile 写入另一个 Shared buffer.
#pragma unroll
        for(int load_idx=0; load_idx<A_LOADS_PER_THREAD; ++load_idx){
            const int s = load_idx * thread_a_stride;
            const float4 tmp = A_global_reg[load_idx];
            As[smem_write][thread_a_col * 4][thread_a_row + s] = tmp.x;
            As[smem_write][thread_a_col * 4 + 1][thread_a_row + s] = tmp.y;
            As[smem_write][thread_a_col * 4 + 2][thread_a_row + s] = tmp.z;
            As[smem_write][thread_a_col * 4 + 3][thread_a_row + s] = tmp.w;
        }
#pragma unroll
        for(int load_idx=0; load_idx<B_LOADS_PER_THREAD; ++load_idx){
            const int s = load_idx * thread_b_stride;
            FLOAT4(Bs[smem_write][thread_b_row + s][thread_b_col * 4]) = B_global_reg[load_idx];
        }

        __syncthreads();
        /* ping pong smem2reg end*/
    }

    // 计算最后一个 tile
    /* ping pong smem2reg start */

#pragma unroll
    // As共享内存 到 A_reg寄存器
    for (int resIdxM=0; resIdxM<TM; resIdxM+=4)
        FLOAT4(A_reg[0][resIdxM]) = FLOAT4(As[smem_write][0][thread_c_row * TM + resIdxM]);
#pragma unroll
    // Bs共享内存 到 B_reg寄存器
    for (int resIdxN=0; resIdxN<TN; resIdxN+=4)
        FLOAT4(B_reg[0][resIdxN]) = FLOAT4(Bs[smem_write][0][thread_c_col * TN + resIdxN]);

    // 一个线程处理 TM 行 TN 列的 C 元素
#pragma unroll
    for (int dotIdx=1; dotIdx<BK; ++dotIdx){
#pragma unroll
        // As共享内存 到 A_reg寄存器
        for (int resIdxM=0; resIdxM<TM; resIdxM+=4){
            FLOAT4(A_reg[dotIdx % 2][resIdxM]) = FLOAT4(As[smem_write][dotIdx][thread_c_row * TM + resIdxM]);
        }
#pragma unroll
        // Bs共享内存 到 B_reg寄存器
        for (int resIdxN=0; resIdxN<TN; resIdxN+=4){
            FLOAT4(B_reg[dotIdx % 2][resIdxN]) = FLOAT4(Bs[smem_write][dotIdx][thread_c_col * TN + resIdxN]);
        }
#pragma unroll
        // 计算结果
        for (int resIdxM=0; resIdxM<TM; ++resIdxM){
            for (int resIdxN=0; resIdxN<TN; ++resIdxN){
                res_reg[resIdxM][resIdxN] += A_reg[(dotIdx - 1) % 2][resIdxM] * B_reg[(dotIdx - 1) % 2][resIdxN];
            }
        }
    }
#pragma unroll
    // 计算最后一个block
    for (int resIdxM=0; resIdxM<TM; ++resIdxM){
        for (int resIdxN=0; resIdxN<TN; ++resIdxN){
            res_reg[resIdxM][resIdxN] += A_reg[(BK - 1) % 2][resIdxM] * B_reg[(BK - 1) % 2][resIdxN];
        }
    }
    /* ping pong smem2reg end*/

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
