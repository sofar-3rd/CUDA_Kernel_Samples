cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 90)
endif()

project(LAYERNORM LANGUAGES CXX CUDA)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

include(CTest)

add_library(layernorm_v1 STATIC src/layernorm.cu)
target_include_directories(layernorm_v1 PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(layernorm_v2 STATIC src/layernorm_v2.cu)
target_include_directories(layernorm_v2 PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(layernorm_cub STATIC src/layernorm_cub.cu)
target_include_directories(layernorm_cub PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(layernorm_welford STATIC src/layernorm_welford.cu)
target_include_directories(layernorm_welford PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(layernorm_v1 layernorm_v2 layernorm_cub layernorm_welford PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(BUILD_TESTING)
    add_executable(layernorm_v1_test tests/layernorm_v1_test.cu)
    target_include_directories(layernorm_v1_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(layernorm_v1_test PRIVATE layernorm_v1)

    add_executable(layernorm_v2_test tests/layernorm_v2_test.cu)
    target_include_directories(layernorm_v2_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(layernorm_v2_test PRIVATE layernorm_v2)

    add_executable(layernorm_cub_test tests/layernorm_cub_test.cu)
    target_include_directories(layernorm_cub_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(layernorm_cub_test PRIVATE layernorm_cub)

    add_executable(layernorm_welford_test tests/layernorm_welford_test.cu)
    target_include_directories(layernorm_welford_test PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(layernorm_welford_test PRIVATE layernorm_welford)

    add_test(NAME layernorm_v1_correctness COMMAND layernorm_v1_test)
    add_test(NAME layernorm_v2_correctness COMMAND layernorm_v2_test)
    add_test(NAME layernorm_cub_correctness COMMAND layernorm_cub_test)
    add_test(NAME layernorm_welford_correctness COMMAND layernorm_welford_test)
endif()
