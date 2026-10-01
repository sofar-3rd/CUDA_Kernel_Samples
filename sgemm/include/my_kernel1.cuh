#pragma once
#include <stdio.h>

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void my_sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
