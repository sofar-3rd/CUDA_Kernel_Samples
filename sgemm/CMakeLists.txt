cmake_minimum_required(VERSION 3.16)
project(CUDA_SGEMM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
list(REMOVE_ITEM SRC
    "${PROJECT_SOURCE_DIR}/src/my_kernel2.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel2_BM_BN.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel3.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel4.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel5.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel6.cu"
    "${PROJECT_SOURCE_DIR}/src/my_kernel7.cu"
)

add_executable(main main.cu ${SRC})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>)

# NOTE: -static-global-template-stub 是 CUDA 11.x 的 nvcc 选项，CUDA 12 已移除。
# 保留会导致该选项被当作未知选项转发给 host compiler(gcc) 而报错。
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>)
endif()

# link cublas library
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

add_executable(my_kernel2_test src/my_kernel2.cu)
set_target_properties(my_kernel2_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel2_test ${CUDA_cublas_LIBRARY})
enable_testing()
add_test(NAME my_kernel2 COMMAND my_kernel2_test)

add_executable(my_kernel2_BM_BN_test src/my_kernel2_BM_BN.cu ${SRC})
set_target_properties(my_kernel2_BM_BN_test PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(my_kernel2_BM_BN_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>)
target_link_libraries(my_kernel2_BM_BN_test ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel2_BM_BN COMMAND my_kernel2_BM_BN_test)

add_executable(my_kernel3_test src/my_kernel3.cu)
set_target_properties(my_kernel3_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel3_test ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel3 COMMAND my_kernel3_test)

add_executable(my_kernel4_test src/my_kernel4.cu)
set_target_properties(my_kernel4_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel4_test ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel4 COMMAND my_kernel4_test)

add_executable(my_kernel5_test src/my_kernel5.cu)
set_target_properties(my_kernel5_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel5_test ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel5 COMMAND my_kernel5_test)

add_executable(my_kernel6_test src/my_kernel6.cu)
set_target_properties(my_kernel6_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel6_test ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel6 COMMAND my_kernel6_test)

add_executable(my_kernel7_test src/my_kernel7.cu)
set_target_properties(my_kernel7_test PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(my_kernel7_test ${CUDA_cublas_LIBRARY})
add_test(NAME my_kernel7 COMMAND my_kernel7_test)
