#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <float.h>
#include "include/utils.cuh"

void check_result(float* host_ref, float* dev_ref, const int N){
    double var = 1e-8;
    int flag = 1;
    
    for(int i=1; i<N; i++){
        if (abs(host_ref[i] - dev_ref[i]) > var){
            flag = 0;
            break;
        }
    }
    if (flag == 0)
        printf("WRONG NOT MATCH!\n");
    else
        printf("MATCH\n");
}

// computer on cpu
void tensor_add_cpu(float* a, float* b, float* output, const int N){
    for (int i=0; i<N; i++){
        output[i] = a[i] + b[i];
    }
}

__global__ void tensor_add_gpu(float* a, float* b, float* output, const int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = a[idx] + b[idx];
}

void initialize_data(float* a, const int N){
    for (int i=0; i<N; ++i){
        a[i] = i;
    }
}

int main(){
    cudaSetDevice(0);
    int N = 1024 * 1024;
    size_t nBytes = N * sizeof(float);

    float *h_a, *h_b, *h_output, *ground_truth;
    float *d_a, *d_b, *d_output;

    // malloc host memory
    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    h_output = (float*)malloc(nBytes);
    ground_truth = (float*)malloc(nBytes);

    initialize_data(h_a, N);
    initialize_data(h_b, N);

    // malloc device global memory
    cudaMalloc((float**)(&d_a), nBytes);
    cudaMalloc((float**)(&d_b), nBytes);
    cudaMalloc((float**)(&d_output), nBytes);

    // copy memory from cpu to gpu
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // initialize grid block size
    dim3 block(64);
    dim3 grid(N/64);

    // computing on gpu
    printf("Begin computing on gpu...\n");
    auto launch = [&] {
        tensor_add_gpu<<<grid, block>>>(d_a, d_b, d_output, N);
    };
    float kernel_time = TIME_RECORD(10, launch);
    printf("kernel time (avg of 10 runs): %.6f ms\n", kernel_time / 10);

    // computing on cpu
    auto t0 = std::chrono::steady_clock::now();
    tensor_add_cpu(h_a, h_b, ground_truth, N);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("cpu time: %.6f ms\n", cpu_ms);

    // copy result memory from gpu to cpu
    cudaMemcpy(h_output, d_output, nBytes, cudaMemcpyDeviceToHost);

    check_result(ground_truth, h_output, N);

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);

    // free cpu memory
    free(h_a);
    free(h_b);
    free(h_output);

    return 0;
}