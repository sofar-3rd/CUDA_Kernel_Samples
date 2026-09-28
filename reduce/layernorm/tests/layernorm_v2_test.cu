#include "layernorm_v2.cuh"
#include "tests/layernorm_test_support.cuh"

int main(int argc, char** argv) {
    constexpr layernorm_test::TestCase test_cases[] = {
        {"vec4-large-M", 1024, 2048, 4,  256},
        {"vec2-path",      33, 1026, 2,  512},
        {"vec1-path",      33, 1025, 1, 1024},
        {"small-N",       257,  256, 4,   64},
    };
    return layernorm_test::run_main(
        argc,
        argv,
        "v2",
        test_cases,
        sizeof(test_cases) / sizeof(test_cases[0]),
        layernorm_v2::choose_launch_config,
        layernorm_v2::launch);
}
