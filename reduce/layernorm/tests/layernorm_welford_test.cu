#include "layernorm_welford.cuh"
#include "tests/layernorm_test_support.cuh"

bool test_invalid_launch() {
    using layernorm_welford::launch;
    if (launch(nullptr, nullptr, nullptr, nullptr, 0, 4, 1e-5f) != cudaErrorInvalidValue
        || launch(nullptr, nullptr, nullptr, nullptr, 1, 0, 1e-5f) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "empty shape must be rejected\n");
        return false;
    }
    float* buffer = nullptr;
    if (!layernorm_test::check_cuda(cudaMalloc(&buffer, 32 * sizeof(float)), "allocate alignment test")) {
        return false;
    }
    bool ok = true;
    for (int shifted = 0; shifted < 4; ++shifted) {
        float* pointers[] = {buffer, buffer + 8, buffer + 16, buffer + 24};
        ++pointers[shifted];
        const auto error = launch(pointers[0], pointers[1], pointers[2], pointers[3], 1, 4, 1e-5f);
        if (error != cudaErrorInvalidValue) {
            std::fprintf(stderr, "unaligned pointer %d must be rejected before launch\n", shifted);
            ok = false;
            break;
        }
    }
    cudaFree(buffer);
    return ok;
}

int main(int argc, char** argv) {
    constexpr layernorm_test::TestCase test_cases[] = {
        {"vec4-large-M", 1024, 2048, 4, 256},
        {"vec2-partial", 33, 1026, 2, 513},
        {"vec1-capped", 33, 1025, 1, 1024},
        {"partial-warp", 33, 255, 1, 255},
        {"small-N", 257, 256, 4, 64},
        {"one-value", 3, 1, 1, 1},
        {"two-values", 3, 2, 2, 1},
        {"four-values", 3, 4, 4, 1},
        {"long-row", 8, 16384, 4, 1024},
    };
    const int result = layernorm_test::run_main(
        argc, argv, "welford", test_cases,
        sizeof(test_cases) / sizeof(test_cases[0]),
        layernorm_welford::choose_launch_config, layernorm_welford::launch);
    if (argc == 1 && result == EXIT_SUCCESS) {
        return test_invalid_launch() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return result;
}
