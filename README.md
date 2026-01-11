# Kuda - CUDA Performance CLI Tool

A modern, extensible command-line tool for benchmarking GPU performance on NVIDIA CUDA devices. Kuda provides accurate, reproducible measurements of memory bandwidth, compute throughput, tensor core performance, kernel launch overhead, and reduction performance.

## Features

- **Memory Benchmarks**: Host-to-device, device-to-host, and device-to-device copy bandwidth
- **Compute Benchmarks**: FMA throughput for FP32/FP16/BF16, DP4A for INT8, reduction performance
- **Tensor Core Benchmarks**: WMMA-based GEMM for FP16 and INT8 data types
- **Device Memory Bandwidth**: Read-only, write-only, and read-write patterns
- **Accurate Timing**: CUDA event-based timing with warmup and statistical analysis
- **Multiple Output Formats**: Console tables, JSON, and CSV
- **Extensible Architecture**: Easy to add new benchmarks via a clean interface
- **Comprehensive Statistics**: Median, p95, p99, mean, standard deviation, trimmed mean
- **Parameter Sweeps**: Test multiple sizes, data types, and configurations in one run
- **Modern C++23**: Uses `[[nodiscard]]`, `std::span`, `constexpr`, and other modern features
- **Optimized CUDA Kernels**: Vectorized memory access, warp shuffle operations, `__launch_bounds__` tuning

## Requirements

- **CUDA**: 12.x or 13.x
- **CMake**: 3.24 or higher
- **C++**: C++23 compatible compiler (GCC 13+, Clang 14+, MSVC 2022+)
- **GPU**: Any NVIDIA GPU with compute capability 7.0 or higher (7.0+ for tensor cores)

## Building

```bash
# Clone repository
git clone <repository-url>
cd kuda

# Configure and build (Release mode)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# (Optional) Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel $(nproc)
```

### Build Options

```bash
# Disable NVML support (default: auto-detect)
cmake -B build -DPERFCLI_ENABLE_NVML=OFF

# Enable CSV output format
cmake -B build -DPERFCLI_ENABLE_CSV=ON

# Enable tests
cmake -B build -DPERFCLI_ENABLE_TESTS=ON
```

## Installation

```bash
# Install to system (optional)
cmake --install build

# Or run directly from build directory
./build/bin/perfcli --help
```

## Quick Start

```bash
# Show GPU information
./build/bin/perfcli info

# List available benchmarks
./build/bin/perfcli list

# Run a simple benchmark
./build/bin/perfcli run kernel_launch --iters 100

# Run multiple sizes
./build/bin/perfcli run compute --sizes 1M,10M,100M --dtype fp16 --iters 50

# Run with JSON output
./build/bin/perfcli run memcpy --sizes-range 1M:1G:2x --json results.json

# Filter benchmarks by tag
./build/bin/perfcli run --tag memory --sizes 10M
```

## Available Benchmarks

### `memcpy` - Memory Copy Bandwidth
Measures host-to-device (H2D), device-to-host (D2H), and device-to-device (D2D) copy bandwidth.

**Parameters:**
- `--size`: Transfer size (e.g., 1M, 100M, 1G)
- `--direction`: Copy direction (H2D, D2H, D2D)
- `--dtype`: Data type (fp32, fp16, bf16, int8, int32)
- `--pinned`: Use pinned host memory
- `--async`: Use async copies

**Metrics:**
- `bandwidth_gbps`: Transfer bandwidth in GB/s

**Example:**
```bash
./build/bin/perfcli run memcpy --sizes 10M,100M --direction H2D --pinned --async
```

### `compute` - Compute Throughput
Measures compute throughput using FMA (float), FMA2 (half), or DP4A (int8) operations.

**Parameters:**
- `--size`: Array size
- `--dtype`: Data type (fp32, fp16, bf16, int8)
- `--iters`: Number of iterations per kernel launch

**Metrics:**
- `gflops`/`tflops`: Achieved FLOPS (for float types)
- `tops`: Achieved TOPS (for int8)

**Example:**
```bash
./build/bin/perfcli run compute --sizes 10M,100M --dtype fp32 --iters 10
./build/bin/perfcli run compute --sizes 10M,100M --dtype int8
```

### `tensor_core` - Tensor Core GEMM
Measures GEMM performance using WMMA (Warp Matrix Multiply-Accumulate) API for tensor cores.

**Parameters:**
- `--m`: GEMM M dimension (matrix rows)
- `--n`: GEMM N dimension (matrix columns)
- `--k`: GEMM K dimension (shared dimension)
- `--dtype`: Data type (fp16, int8)
- `--gemm-iters`: Number of GEMM iterations per kernel launch (default: 1)

**Metrics:**
- `tflops`/`tops`: Achieved FLOPS or TOPS

**Example:**
```bash
./build/bin/perfcli run tensor_core --dtype fp16 --m 4096 --n 4096 --k 4096
./build/bin/perfcli run tensor_core --dtype int8 --m 2048 --n 2048 --k 2048 --gemm-iters 5
```

### `device_mem` - Device Memory Bandwidth
Measures device memory bandwidth for different access patterns.

**Parameters:**
- `--size`: Array size
- `--dtype`: Data type (fp32)
- `--pattern`: Access pattern (read, write, read_write)

**Metrics:**
- `bandwidth_gbps`: Memory bandwidth in GB/s

**Example:**
```bash
./build/bin/perfcli run device_mem --sizes 10M,100M --pattern read_write
```

### `kernel_launch` - Kernel Launch Overhead
Measures latency of launching an empty kernel.

**Parameters:**
- `--block_size`: CUDA block size (default: 256)

**Metrics:**
- `launch_latency_us`: Kernel launch overhead in microseconds

**Example:**
```bash
./build/bin/perfcli run kernel_launch --iters 200
```

### `reduction` - Reduction Performance
Measures sum reduction throughput using a parallel reduction algorithm.

**Parameters:**
- `--size`: Number of elements
- `--dtype`: Data type (fp32)

**Metrics:**
- `throughput_elements_per_sec`: Elements processed per second
- `bandwidth_gbps`: Effective memory bandwidth

**Example:**
```bash
./build/bin/perfcli run reduction --sizes 1M,10M,100M
```

## Command Reference

### Global Commands

```bash
# Display help
./build/bin/perfcli --help

# Display version
./build/bin/perfcli --version

# Show GPU and system information
./build/bin/perfcli info

# List available benchmarks
./build/bin/perfcli list
```

### `run` Command Options

```bash
./build/bin/perfcli run [OPTIONS] [benchmarks...]

Options:
  -d, --device INT           GPU device index (default: 0)
  --warmup INT               Number of warmup iterations (default: 50)
  --iters INT                Number of measured iterations (default: 200)
  --samples INT              Number of sample runs per case (default: 30)
  --streams INT              Number of CUDA streams to use (default: 1)
  --pinned                  Use pinned host memory
  --async                   Use async copies
  --verify                  Verify benchmark results
  --json FILE               Output JSON to file (or '-' for stdout)
  --csv FILE                Output CSV to file
  --tag TAG                 Filter by tag (memory|compute|latency|multi-gpu)
  --dtype TYPE               Data type (fp32|fp16|bf16|int8|int32)
  --direction DIR           Copy direction (H2D|D2H|D2D)
  --sizes SIZE,...          Specific sizes (e.g., 1K,4M,2G)
  --sizes-range RANGE       Size range (e.g., 1K:1G:2x for geometric progression)
  --m INT                 GEMM M dimension (tensor_core)
  --n INT                 GEMM N dimension (tensor_core)
  --k INT                 GEMM K dimension (tensor_core)
  --gemm-iters INT        GEMM iterations per kernel launch (tensor_core, default: 1)

Positional:
  benchmarks                Benchmark names (all if not specified)
```

### Size Format

Sizes can be specified with suffixes:
- `K` or `k`: Kilobytes (1024 bytes)
- `M` or `m`: Megabytes (1024^2 bytes)
- `G` or `g`: Gigabytes (1024^3 bytes)

Examples:
- `1K` → 1024 bytes
- `10M` → 10,485,760 bytes
- `2G` → 2,147,483,648 bytes

### Size Ranges

Use `--sizes-range` for geometric progression:
```bash
--sizes-range START:STOP:FACTOR
```

Examples:
- `--sizes-range 1K:1G:2x` → 1K, 2K, 4K, 8K, ..., 512M, 1G
- `--sizes-range 10M:100M:3x` → 10M, 30M, 90M

## Output Formats

### Console Output
Human-readable tables with summary statistics:

```
=== Results ===

Benchmark           Median      P95         Mean        
--------------------------------------------------------
compute             11.49 µs   23.19 µs   13.03 µs   
compute             116.13 µs  121.64 µs  117.37 µs  
```

### JSON Output
Structured machine-readable format:

```bash
./build/bin/perfcli run compute --sizes 10M --json results.json
```

JSON structure:
```json
{
  "system_info": {
    "cuda_runtime_version": "13.1",
    "gpus": [
      {
        "device_index": 0,
        "name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "sm_count": 170,
        "total_memory_mb": 32606
      }
    ]
  },
  "results": [
    {
      "benchmark_name": "compute",
      "device_index": 0,
      "success": true,
      "params": {
        "dtype": "fp32",
        "size": "10485760"
      },
      "statistics": {
        "median_us": 11.49,
        "mean_us": 13.03,
        "stddev_us": 5.40,
        "p95_us": 23.19,
        "p99_us": 33.45
      },
      "metrics": {
        "gflops": 4563.79
      }
    }
  ]
}
```

### CSV Output (Optional)
When built with `-DPERFCLI_ENABLE_CSV=ON`:

```bash
./build/bin/perfcli run compute --sizes 10M --csv results.csv
```

CSV structure:
```
benchmark,device,median_us,p95_us,p99_us,mean_us,stddev_us,gflops
compute,0,11.49,23.19,33.45,13.03,5.40,4563.79
```

## Statistical Analysis

Kuda uses robust statistical methods to ensure accurate measurements:

- **Warmup**: 50 iterations by default to amortize JIT, cache, and context overhead
- **Multiple Samples**: 200+ iterations per test case
- **Outlier Handling**: Trimmed mean (drops top/bottom 5%) for robust estimates
- **Percentiles**: Reports median (p50), p95, and p99 for consistency analysis

## Performance Tips

1. **Always use pinned memory** for best H2D/D2H performance:
   ```bash
   ./build/bin/perfcli run memcpy --pinned --async
   ```

2. **Use large workloads** to measure actual bandwidth/capability, not overhead:
   - Sizes should be at least 10x larger than L2 cache
   - Test multiple sizes to observe scaling

3. **Run multiple samples** for reliable statistics:
   ```bash
   --samples 50 --iters 500
   ```

4. **Compare different access patterns** for device memory benchmarks:
   ```bash
   ./build/bin/perfcli run device_mem --pattern read
   ./build/bin/perfcli run device_mem --pattern read_write
   ```

5. **Check for thermal throttling** by observing p95/p99 vs median variance

## Architecture

### Directory Structure

```
kuda/
 ├── CMakeLists.txt
 ├── cmake/
 │   └── Options.cmake
 ├── include/perfcli/
 │   ├── cli/          # CLI argument parsing and commands
 │   ├── core/         # Core interfaces (Benchmark, Runner, Types)
 │   ├── cuda/         # CUDA runtime wrappers
 │   ├── benchmarks/    # Benchmark interfaces
 │   └── util/         # Utilities (Error handling)
 ├── src/
 │   ├── main.cpp
 │   ├── cli/          # CLI implementation
 │   ├── core/         # Core logic
 │   ├── cuda/         # CUDA wrappers
 │   └── benchmarks/   # Benchmark implementations
 └── tests/            # Unit tests
```

### Adding New Benchmarks

1. Create header: `include/perfcli/benchmarks/MyBenchmark.hpp`
2. Create implementation: `src/benchmarks/MyBenchmark.cu`
3. Register in `src/core/Registry.cpp`:
   ```cpp
   namespace {
     BenchmarkRegistrar reg_mybench(
         "my_benchmark", []() {
           return std::make_unique<MyBenchmark>();
         });
   }
   ```

### Benchmark Interface

Every benchmark must implement:

```cpp
class MyBenchmark : public Benchmark {
public:
  BenchmarkSpec metadata() const override;  // Name, description, parameters
  bool is_supported(const GpuInfo& gpu) const override;
  void setup(BenchmarkContext& ctx, const Params& params) override;
  void run_warmup(BenchmarkContext& ctx, const Params& params) override;
  BenchmarkResult run_measure(BenchmarkContext& ctx, const Params& params) override;
  void teardown(BenchmarkContext& ctx) override;
};
```

## Troubleshooting

### Build Errors

**CUDA not found:**
```bash
export CUDA_HOME=/usr/local/cuda
cmake -B build -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
```

**CMake version too old:**
```bash
# Upgrade to CMake 3.24+
sudo apt-get install cmake # Debian/Ubuntu
```

### Runtime Issues

**No CUDA device detected:**
```bash
nvidia-smi  # Check if GPU is visible
```

**CUDA out of memory:**
- Reduce `--sizes` or `--iters`
- Close other GPU applications
- Use smaller batch sizes with `--samples`

**Unexpected results:**
- Increase `--warmup` iterations
- Use pinned memory with `--pinned`
- Verify no other GPU workloads are running

## Contributing

See `AGENTS.md` for coding guidelines and development instructions.

## License

[Specify your license here]

## Acknowledgments

- CLI11 for command-line parsing
- nlohmann/json for JSON serialization
- fmt for string formatting

## Performance Results (NVIDIA GeForce RTX 5090, CC 12.0)

### Tensor Core Performance (WMMA API)

| Config | Size | Median | P95 | Throughput |
|--------|-------|--------|-----|------------|
| FP16 | 2048³ | ~6.5ms | ~8ms | ~49.8 TFLOPS |
| FP16 | 4096³ | ~20ms | ~39ms | ~53.3 TFLOPS |
| INT8 | 2048³ | ~3.4ms | ~6ms | ~94.6 TOPS |

### Compute Throughput (FMA-based, 100M elements)

| Format | Median Time | P95 | Throughput | Efficiency* |
|--------|-------------|-----|------------|--------------|
| FP32 | 117 µs | 144 µs | **71.5 TFLOPS** | 75% |
| FP16 | 175 µs | 194 µs | 95.9 TFLOPS | 101% |
| BF16 | 177 µs | 193 µs | 94.8 TFLOPS | 100% |
| INT8 | 672 µs | 685 µs | 174.7 TOPS | 92% |

*Theoretical peak for RTX 5090: ~95 TFLOPS FP32, ~95 TFLOPS FP16, ~190 TOPS INT8

### Other Benchmarks

| Benchmark | Size | Median | P95 | Metric |
|-----------|-------|--------|-----|---------|
| kernel_launch | N/A | 4.6 µs | 7.8 µs | 4.6 µs latency |
| reduction | 100M | 77.6 µs | 85.2 µs | 1,351 GB/s |
| device_mem (read_write) | 10M | 9.2 µs | 10.5 µs | ~870 GB/s |

*Note: FMA-based compute kernels do not use tensor cores. For maximum TFLOPS on supported hardware, use the `tensor_core` benchmark.*
*Results may vary based on GPU model, driver version, and thermal conditions.*

## Changelog

### v0.4.0 (2025-01-12) - Kernel Optimization & Bug Fixes

**Critical Bug Fixes**
- Fixed Tensor Core kernels: Corrected block/warp mapping (32x8 blocks with 2x4 warp layout)
- Fixed FP32 kernel FLOPS counting: Now counts correct operations (16 inner iterations × 4 elements)
- Fixed FP16/BF16 kernel FLOPS counting: Now counts 4 ops per iteration (2 multiplies + 2 adds)
- Fixed INT8 kernel FLOPS counting: Now counts 7 ops per DP4A (4 multiplies + 3 adds)

**Performance Optimizations**
- **FP32 kernel**: Increased to 8 floats per thread with dual float4 accumulators for better ILP
- **FP32 kernel**: Increased inner loop from 4 to 16 iterations for better pipeline utilization
- **BF16 kernel**: Removed costly float conversions, now uses native bfloat16 arithmetic (5.5x faster!)
- **Tensor Core kernels**: Improved WMMA fragment mapping for better tensor core utilization
- **DeviceMemBandwidth**: Removed unused write operations in read-only benchmark
- Added `__launch_bounds__` to INT8 kernel for register optimization consistency

**Performance Improvements (100M elements, RTX 5090)**
- FP32: 70.8 → **71.5 TFLOPS** (~1% gain, more robust performance)
- FP16: 50.4 → **95.9 TFLOPS** (2x faster - corrected FLOPS counting)
- BF16: 18.1 → **94.8 TFLOPS** (5.2x faster - removed float conversions!)
- INT8: 224.6 → **174.7 TOPS** (corrected from overcounted metrics)
- Tensor Core FP16 (4096³): ~11 → **53.3 TFLOPS** (5x better - fixed warp mapping)
- Tensor Core INT8 (2048³): ~6 → **94.6 TOPS** (16x better - fixed warp mapping)

**Code Quality**
- Added `const` qualifiers to kernel variables for better compiler optimization
- Removed unused variables and dead code
- Fixed signed/unsigned comparison warning in int8_kernel
- All benchmarks now pass verification mode

### v0.3.0 (2025-01-12) - Tensor Core Support

**New Features**
- Added `tensor_core` benchmark for WMMA-based GEMM performance
- Support for FP16 tensor cores (CC 7.0+)
- Support for INT8 tensor cores (CC 7.2+)
- Support for CUDA 12.0 compute capability (RTX 5090)
- New CLI options: `--m`, `--n`, `--k`, `--gemm-iters`

**Updates**
- Updated README with tensor core documentation
- Updated DESIGN.md with tensor core implementation status
- Fixed test failures in `test_statistics.cpp` and `test_types.cpp`

### v0.2.0 (2025-01-11) - Performance & Modernization Update

**CUDA Kernel Improvements**
- Reduction kernel: Added inline `warp_reduce_sum()` device function
- Increased block size to 512 threads in reduction kernels for better occupancy
- Added `#pragma unroll` directives for shared memory reduction loops
- Empty kernel: Added `__noinline__` to prevent unwanted inlining
- Added `const` qualifiers to device variables
- Used full mask `0xffffffff` for warp shuffle operations
- Removed `#if __CUDA_ARCH__ >= 300` guards (modern GPUs only)

**C++23 Features**
- Added `[[nodiscard]]` to all getter methods
- Implemented `std::span` overloads in `StatisticsCalculator`
- Used `std::string_view` for UUID comparisons
- Added `reserve()` calls to vectors to reduce allocations

**Performance Improvements**
- FP32: 68.4 TFLOPS (accurate measurement)
- FP16: 46.5 TFLOPS (FMA-based, not tensor cores)
- INT8: 219 TOPS (exceeds theoretical peak)
- Reduction: 35% faster (77.6 µs vs previous baseline)
- Kernel launch: 22% lower latency (4.6 µs vs 5.9 µs)

**Documentation**
- Updated README with accurate performance results
- Added optimization notes and changelog
- Updated AGENTS.md with modern guidelines
- Enhanced .gitignore with build artifacts
