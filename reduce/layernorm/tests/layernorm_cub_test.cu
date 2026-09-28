#include "layernorm_cub.cuh"
#include "tests/layernorm_test_support.cuh"

int main(int argc, char** argv) {
    constexpr layernorm_test::TestCase test_cases[] = {
        {"vec4-large-M", 1024, 2048, 4,  256},
        {"vec2-partial",   33, 1026, 2,  513},
        {"vec1-capped",    33, 1025, 1, 1024},
        {"partial-warp",   33,  255, 1,  255},
        {"small-N",       257,  256, 4,   64},
    };
    return layernorm_test::run_main(
        argc,
        argv,
        "cub",
        test_cases,
        sizeof(test_cases) / sizeof(test_cases[0]),
        layernorm_cub::choose_launch_config,
        layernorm_cub::launch);
}
