# LayerNorm CUDA

这个目录包含三个 FP32 LayerNorm 前向实现：

- `layernorm_v1`：固定 128 threads、标量访存的基线实现。
- `layernorm_v2`：动态 block size 与 `vec_size=1/2/4` 访存练习。
- `layernorm_cub`：保留 v2 结构、改用 CUB BlockReduce 的规约练习。

## 目录结构

```text
layernorm/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── layernorm.cuh
│   ├── layernorm_v2.cuh
│   ├── layernorm_cub.cuh
│   └── utils.cuh
├── src/
│   ├── layernorm.cu
│   ├── layernorm_v2.cu
│   ├── layernorm_cub.cu
│   └── utils.cu
└── tests/
    ├── layernorm_test_support.cuh
    ├── layernorm_v1_test.cu
    ├── layernorm_v2_test.cu
    └── layernorm_cub_test.cu
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
./layernorm_cub_test --benchmark
```

以上命令运行默认 shape sweep。也可以显式指定一个 shape，适合精确 benchmark 和 NCU 采集：

```bash
./layernorm_v1_test --benchmark 1024 2048
./layernorm_v2_test --benchmark 1024 2048
./layernorm_cub_test --benchmark 1024 2048
```

`M` 和 `N` 必须是正整数。

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

## CUB BlockReduce 实现

`src/layernorm_cub.cu` 保留 v2 的向量化和动态 launch 结构，使用 CUB `BlockReduce` 完成 block 级规约：

```cpp
block_reduce_sum(
    float value,
    CubBlockReduce::TempStorage& reduce_storage,
    int valid_threads)
```

编译和运行：

```bash
make layernorm_cub_test -j
./layernorm_cub_test
./layernorm_cub_test --benchmark
```

正确性测试包含 `block=255` 和 `block=513`，用于覆盖 partial warp 和非二次幂 block。

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

sudo ncu \
  --target-processes all \
  --kernel-name 'regex:layernorm' \
  --launch-skip 10 \
  --launch-count 1 \
  --set full \
  --export layernorm-cub \
  --force-overwrite \
  ./build/layernorm_cub_test --benchmark 4096 8192


# 注意：不能在 --metrics 的单个参数值内部用 `\` 续行。
# `\` 只吃掉换行符，下一行的缩进空格会留在值里，ncu 会把
# "   dram__throughput..." 当成目标程序，报 "does not exist or is not an executable"。
# 用 tr 去掉所有空白后再拼成一个逗号列表：
METRICS="$(tr -d '[:space:]' <<'EOF'
  gpu__time_duration.sum,
  dram__throughput.avg.pct_of_peak_sustained_elapsed,
  sm__throughput.avg.pct_of_peak_sustained_elapsed,
  smsp__warps_eligible.avg.per_cycle_active,
  sm__maximum_warps_avg_per_active_cycle,
  sm__warps_active.avg.per_cycle_active
EOF
)"

# 不同 option 之间用 `\` 续行是安全的（空格是 shell 正常分词）
ncu --metrics "$METRICS" \
  --launch-skip 10 \
  --launch-count 1 \
  ./build/layernorm_cub_test --benchmark 4096 8192

ncu --import ./ncu_profile/layernorm-cub.ncu-repz --page raw > ./ncu_profile/layernorm-cub-raw.txt

ncu --import ./ncu_profile/layernorm-cub.ncu-repz --page detail > ./ncu_profile/layernorm-cub-details.txt

ncu --import ./ncu_profile/layernorm-cub.ncu-repz --page source > ./ncu_profile/layernorm-cub-source.txt
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
