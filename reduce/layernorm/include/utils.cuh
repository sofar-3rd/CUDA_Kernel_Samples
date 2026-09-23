#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define cudaCheck(error) _cudaCheck((error), __FILE__, __LINE__)

// 执行一次预热, 再记录 N 次 CUDA 操作的总耗时, 单位为毫秒.
#define TIME_RECORD(N, function)                                                \
    [&] {                                                                       \
        float total_time = 0.0f;                                                \
        for (int repeat = 0; repeat <= (N); ++repeat) {                         \
            cudaEvent_t start;                                                  \
            cudaEvent_t stop;                                                   \
            cudaCheck(cudaEventCreate(&start));                                 \
            cudaCheck(cudaEventCreate(&stop));                                  \
            cudaCheck(cudaEventRecord(start));                                  \
            function();                                                         \
            cudaCheck(cudaEventRecord(stop));                                   \
            cudaCheck(cudaEventSynchronize(stop));                              \
            float elapsed_time = 0.0f;                                          \
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));         \
            if (repeat > 0) {                                                   \
                total_time += elapsed_time;                                     \
            }                                                                   \
            cudaCheck(cudaEventDestroy(start));                                 \
            cudaCheck(cudaEventDestroy(stop));                                  \
        }                                                                       \
        return total_time;                                                      \
    }()

void _cudaCheck(cudaError_t error, const char* file, int line);
