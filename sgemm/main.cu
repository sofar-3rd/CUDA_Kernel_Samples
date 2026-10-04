#include <stdio.h>
#include "utils.cuh"
#include "include/utils.cuh"

int main(int argc, char **argv) {
    // 0. get kernel num: cuBLAS(0) or custom kernel(>0)
    if (argc != 2) {
        printf("Please select a kernel: cuBLAS(0) or custom kernel(>0).\n");
        exit(EXIT_FAILURE);
    }

    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || (kernel_num > 7 && kernel_num < 11) || kernel_num > 17) {
        printf("Please enter a valid kernel number (0-7 or 11-17).\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Select kernel %d.\n", kernel_num);
    };

    // 1. init param
    size_t size_len = 10;  // modify it according to your device. 
    size_t SIZE[size_len];
    for (int i = 0; i < size_len; i++) {
        SIZE[i] = 256 * (i + 1);
    }
    printf("max_size=%lu*%lu\n", SIZE[size_len - 1], SIZE[size_len - 1]);
    const float alpha = 1.0f, beta  = 0.0f; // two arbitary input parameters, C=α*AB+β*C

    // 2. call kernel with different size
    for (int i = 0; i < size_len; i++) {
        size_t m, n, k;
        m = n = k = SIZE[i];
        size_t size_M = m * n;
        size_t mem_size_M = sizeof(float) * size_M;
        
        // 3. host malloc for matrix A, B, C, C_ref(cuBLAS)
        float *h_A = (float *)malloc(mem_size_M);
        float *h_B = (float *)malloc(mem_size_M);
        float *h_C = (float *) malloc(mem_size_M);
        float *h_C_ref = (float *) malloc(mem_size_M);

        // 4. random init elements in A, B, C
        randomize_matrix(h_A, size_M);
        randomize_matrix(h_B, size_M);
        randomize_matrix(h_C, size_M);
        copy_matrix(h_C, h_C_ref, size_M);

        // 5. cuda malloc for A, B, C, C_ref(cuBLAS) and copy A, B from host to device
        float *d_A, *d_B, *d_C, *d_C_ref;
        cudaCheck(cudaMalloc((void **) &d_A, mem_size_M));
        cudaCheck(cudaMalloc((void **) &d_B, mem_size_M));
        cudaCheck(cudaMalloc((void **) &d_C, mem_size_M));
        cudaCheck(cudaMalloc((void **) &d_C_ref, mem_size_M));
        cudaCheck(cudaMemcpy(d_A, h_A, mem_size_M, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B, h_B, mem_size_M, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_C, h_C, mem_size_M, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_C_ref, h_C_ref, mem_size_M, cudaMemcpyHostToDevice));

        // 6. call custom kernel and cublas, and verify correctness
        if (kernel_num != 0) {
            call_kernel(kernel_num, false, m, n, k, alpha, d_A, d_B, beta, d_C);
            call_kernel(0, false, m, n, k, alpha, d_A, d_B, beta, d_C_ref);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaMemcpy(h_C, d_C, mem_size_M, cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(h_C_ref, d_C_ref, mem_size_M, cudaMemcpyDeviceToHost));
            if (!verify_matrix(h_C, h_C_ref, size_M)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            }
        }
        // 7. time record
        float total_time = call_kernel(kernel_num, true, m, n, k, alpha, d_A, d_B, beta, d_C);;
        total_time /= 1000.;
        printf("m=n=k=%lu\n", m);
        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%lu).\n",
            total_time / REPEAT_TIMES, 2. * 1e-9 * REPEAT_TIMES * m * n * k / total_time, m);
        fflush(stdout);
        
        // 8. free memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        cudaCheck(cudaFree(d_A));
        cudaCheck(cudaFree(d_B));
        cudaCheck(cudaFree(d_C));
        cudaCheck(cudaFree(d_C_ref));
    }
    return 0;
}