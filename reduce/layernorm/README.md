# LayerNorm CUDA

这个目录包含两个 FP32 LayerNorm 前向实现：

- `layernorm_v1`：固定 128 threads、标量访存的基线实现。
- `layernorm_v2`：练习实现，目标是动态选择 block size，并支持 `vec_size=1/2/4` 的访存路径。

## 目录结构

```text
layernorm/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── layernorm.cuh
│   ├── layernorm_v2.cuh
│   └── utils.cuh
├── src/
│   ├── layernorm.cu
│   ├── layernorm_v2.cu
│   └── utils.cu
└── tests/
    ├── layernorm_test_support.cuh
    ├── layernorm_v1_test.cu
    └── layernorm_v2_test.cu
```

## 环境要求

- NVIDIA GPU
- CUDA Toolkit
- CMake 3.18 或更高版本
- 支持 C++17 的编译器

CMake 默认生成以下 GPU 架构的代码：

- `sm_86`：RTX 3060 Ti 等 Ampere GPU
- `sm_90`：H20/H100 等 Hopper GPU

如需其他架构，在配置时覆盖 `CMAKE_CUDA_ARCHITECTURES`。

## 编译

在本目录使用传统 CMake 流程：

```bash
mkdir -p build
cd build
cmake ..
make -j
```

v2 测试目标不参与默认构建。完成练习 TODO 后单独编译：

```bash
make layernorm_v2_test -j
```

如果需要重新配置，可以删除构建目录后重来：

```bash
cd ..
rm -rf build
mkdir build && cd build
cmake ..
make -j
```

指定其他 GPU 架构，例如只编译 H20：

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j
```

## 正确性测试

以下命令均在 `build` 目录执行。

运行 v1：

```bash
ctest -R layernorm_v1_correctness --output-on-failure
```

或者直接运行，以查看每组 shape 的误差：

```bash
./layernorm_v1_test
```

完成并编译 v2 后运行：

```bash
ctest -R layernorm_v2_correctness --output-on-failure
./layernorm_v2_test
```

测试使用 double 累加的 CPU 实现作为 reference，并覆盖：

- v1 的主 shape、奇数 `N` 和小 `N`；
- v2 的 `vec_size=4/2/1` 和动态 block size。

练习 TODO 尚未完成时，`layernorm_v2_test` 的动态配置断言失败是预期的 RED 状态。

## 性能测试

benchmark 不注册为默认 CTest，避免机器负载导致测试不稳定。直接执行：

在 `build` 目录执行：

```bash
./layernorm_v1_test --benchmark
./layernorm_v2_test --benchmark
```

输出示例：

```text
[BENCH] v1 M=1024 N=2048 vec=1 block=128 time=59.021 us effective=284.54 GB/s
```

其中有效带宽使用算法口径：

```text
B_algo = 8 * M * N + 8 * N
BW_effective = B_algo / kernel_time
```

## v2 实现契约

只需修改：

```text
src/layernorm_v2.cu
```

并实现 `include/layernorm_v2.cuh` 声明的接口：

```cpp
layernorm_v2::LaunchConfig choose_launch_config(
    std::size_t M,
    std::size_t N);

cudaError_t launch(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t M,
    std::size_t N,
    float epsilon,
    cudaStream_t stream);
```

当前测试要求的 launch 配置：

| M | N | vector size | block size |
|---:|---:|---:|---:|
| 1024 | 2048 | 4 | 256 |
| 33 | 1026 | 2 | 512 |
| 33 | 1025 | 1 | 1024 |
| 257 | 256 | 4 | 64 |

框架已经提供完整的三阶段 LayerNorm、block reduction、`vec_size=1/2/4` dispatch 和多累加器。你只需完成 `src/layernorm_v2.cu` 中的两个练习：

1. `choose_block_size()`：动态选择 block size；
2. `load_vector<4>()` / `store_vector<4>()`：将标量循环替换为真正的 `float4` 宽访存。

每完成一步都运行 `./layernorm_v2_test`，最后再运行 benchmark 和 ncu。

## 使用 Nsight Compute

先保证正确性测试通过，再采集单次 kernel：

```bash
sudo ncu \
  --target-processes all \
  --kernel-name 'regex:layernorm' \
  --launch-count 1 \
  --set detailed \
  --export layernorm-v2 \
  --force-overwrite \
  ./build/layernorm_v2_test --benchmark
```

重点比较 v1/v2 的：

- `gpu__time_duration.sum`
- `dram__bytes_read.sum` / `dram__bytes_write.sum`
- global load/store requests
- global load/store sectors
- sectors/request
- Long Scoreboard
- eligible warps
- registers/thread
- achieved occupancy
