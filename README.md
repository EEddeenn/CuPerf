# Kuda - CUDA Performance CLI Tool

A modern, extensible command-line tool for benchmarking GPU performance on NVIDIA CUDA devices. Kuda provides accurate, reproducible measurements of memory bandwidth, compute throughput, kernel launch overhead, and reduction performance.

## Features

- **Memory Benchmarks**: Host-to-device, device-to-host, and device-to-device copy bandwidth
- **Compute Benchmarks**: FMA throughput, reduction performance, kernel launch overhead
- **Device Memory Bandwidth**: Read-only, write-only, and read-write patterns
- **Accurate Timing**: CUDA event-based timing with warmup and statistical analysis
- **Multiple Output Formats**: Console tables, JSON, and CSV
- **Extensible Architecture**: Easy to add new benchmarks via a clean interface
- **Comprehensive Statistics**: Median, p95, p99, mean, standard deviation, trimmed mean
- **Parameter Sweeps**: Test multiple sizes, data types, and configurations in one run

## Requirements

- **CUDA**: 12.x or 13.x
- **CMake**: 3.24 or higher
- **C++**: C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2022+)
- **GPU**: Any NVIDIA GPU with compute capability 7.5 or higher

## Building

```bash
# Clone the repository
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
./build/bin/perfcli run compute --sizes 1M,10M,100M --iters 50

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
Measures FMA (fused multiply-add) compute throughput.

**Parameters:**
- `--size`: Array size
- `--dtype`: Data type (fp32, fp16, bf16)
- `--iters`: Number of FMA iterations per kernel launch

**Metrics:**
- `gflops`: Achieved GFLOPS

**Example:**
```bash
./build/bin/perfcli run compute --sizes 10M,100M --iters 10
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
Measures the latency of launching an empty kernel.

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

## Performance Results (NVIDIA GeForce RTX 5090)

| Benchmark | Size | Median | P95 | Metric |
|-----------|-------|--------|-----|---------|
| memcpy H2D (pinned) | 100M | 28.1 µs | 32.5 µs | 355 GB/s |
| compute (fp32) | 100M | 116.1 µs | 121.6 µs | 4,515 GFLOPS |
| kernel_launch | N/A | 5.9 µs | 41.2 µs | N/A |
| device_mem (rw) | 100M | 18.9 µs | 40.7 µs | 10,560 GB/s |
| reduction | 100M | 116.1 µs | 121.6 µs | 895 GB/s |

*Results may vary based on GPU model, driver version, and thermal conditions.*
