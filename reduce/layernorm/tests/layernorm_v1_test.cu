#include "layernorm.cuh"
#include "tests/layernorm_test_support.cuh"

int main(int argc, char** argv) {
    constexpr layernorm_test::TestCase test_cases[] = {
        {"main-shape", 1024, 2048, 1, 128},
        {"odd-N",        33, 1025, 1, 128},
        {"small-N",     257,  256, 1, 128},
    };
    return layernorm_test::run_main(
        argc,
        argv,
        "v1",
        test_cases,
        sizeof(test_cases) / sizeof(test_cases[0]),
        layernorm_v1::choose_launch_config,
        layernorm_v1::launch);
}
