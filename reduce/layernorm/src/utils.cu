#include "utils.cuh"

void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error at %s:%d: %s\n",
            file,
            line,
            cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
