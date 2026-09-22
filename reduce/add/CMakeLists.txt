cmake_minimum_required(VERSION 3.16)
project(ADD LANGUAGES CXX CUDA)

# RTX 3060 Ti is sm_86; emit native cubin so the driver does not need to JIT
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

add_executable(add add.cu ${SRC})
