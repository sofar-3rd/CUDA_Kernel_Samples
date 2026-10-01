cmake_minimum_required(VERSION 3.16)
project(CUDA_SGEMM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

add_executable(main main.cu ${SRC})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>)

# link cublas library
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
