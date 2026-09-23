cmake_minimum_required(VERSION 3.18)

project(LAYERNORM LANGUAGES CXX CUDA)

add_executable(layernorm layernorm.cu)

set_target_properties(layernorm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
