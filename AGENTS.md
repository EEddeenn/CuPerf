# Agent Guidelines for Kuda (CUDA Performance CLI Tool)

## Build System

CMake with C++23 and CUDA 13.1/12.x. Supports compute capabilities: 75, 80, 86, 89, 90, 120.

### Essential Commands

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel $(nproc)

# Build with tests
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPERFCLI_ENABLE_TESTS=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build -R <test_name> -V    # Run single test
ctest --test-dir build -V                   # Run all tests

# Run CLI
./build/bin/perfcli --help
./build/bin/perfcli run memcpy --sizes-range 1M:8G:2x --json results.json
```

### CMake Options

- `PERFCLI_ENABLE_NVML` (default: ON)
- `PERFCLI_ENABLE_CUPTI` (default: OFF)
- `PERFCLI_ENABLE_CSV` (default: OFF)
- `PERFCLI_ENABLE_TESTS` (default: OFF)
- `PERFCLI_ENABLE_STATIC` (default: OFF)

## Code Style Guidelines

### File Organization

- Headers: `include/perfcli/<module>/`
- Sources: `src/<module>/`
- CUDA kernels: `src/benchmarks/*.cu`
- Public interfaces must be in `include/perfcli/`

### Naming Conventions

- Classes/Structs: `PascalCase` (e.g., `BenchmarkResult`, `DeviceBuffer`)
- Functions/Methods: `snake_case` (e.g., `run_warmup`, `create_streams`)
- Member variables: `snake_case_` (trailing underscore for private members)
- Constants: `kPascalCase` (e.g., `kDefaultIterations`)
- Enums: `PascalCase` with `kPascalCase` values
- Files: `PascalCase` for headers, `PascalCase.cpp` for sources, `PascalCase.cu` for kernels

### Imports and Includes

- Group includes in order: system headers, external library headers, project headers
- Use angle brackets for external includes: `#include <cuda_runtime.h>`
- Use quotes for project includes: `#include "perfcli/core/Runner.hpp"`
- Avoid unnecessary includes; use forward declarations where possible

### Formatting

- Use 2-space indentation (C++ standard)
- Max line length: 100 characters
- Place opening braces on same line for functions/classes (K&R style)
- One statement per line
- Space after keywords: `if (condition)`, `for (auto& x : vec)`
- No space between function name and opening paren: `function(args)`
- Spaces around binary operators: `a + b`, `a == b`
- No spaces inside parentheses: `(a, b)`, not `( a, b )`
- Use `[[nodiscard]]` on all functions returning values that must not be ignored (getters, factories, etc.)
- Lambda pattern for parameter extraction: `auto get_param = [&](...) { ... };`

### Type System

Use strong types over primitives (`DeviceIndex` vs `int`). Prefer `std::chrono` for timing, `std::optional` for nullables, `std::expected` (C++23) for errors, `std::span` for array views, `std::format` for string formatting. Use `auto` when type is obvious from context; prefer explicit types otherwise.

### Memory Management

RAII for all resources. `DeviceBuffer<T>` for device memory, `HostBuffer<T>` for host memory (pageable or pinned). Use `std::make_unique`/`std::make_shared`, never raw `new`/`delete`. Follow rule of five for resource-managing classes.

### CUDA-Specific Guidelines

Always `CUDA_CHECK(call)` - throws `CudaError`. Use `CUDA_CHECK_LAST()` after kernel launches to detect launch errors. Use `EventTimer` for timing: `timer.start()` → enqueue → `timer.stop()` → `timer.sync()` → `timer.elapsed_microseconds()`. Avoid `cudaDeviceSynchronize()` in inner loops. Don't measure allocation time or create streams/events in inner loops. Use `inline __device__` for small helper functions. Use `extern __shared__` for dynamic shared memory.

### CUDA Optimization for Modern GPUs (RTX 5090, Compute 12.0+)

- Use `__launch_bounds__(threads, minBlocks)` for register optimization
- Use vectorized loads/stores (`float4`, `half2`, `float4*`) for memory bandwidth
- Use `__restrict__` on pointer parameters to enable compiler optimizations
- Use `constexpr` for compile-time constants
- Use `#pragma unroll` for small, fixed-count loops
- Use warp shuffle instructions (`__shfl_down_sync`) for intra-warp communication instead of shared memory
- Use `__shfl_down_sync` with full mask `0xffffffff` for known active lanes
- Use `__syncthreads()` carefully; avoid when warp shuffle suffices
- Use `inline` on small device functions
- Use `__noinline__` to prevent unwanted inlining of empty kernels
- Check compute capability with `gpu.is_compute_capability_at_least()` for feature gating
- Use `__builtin_assume_aligned` when alignment is guaranteed
- Consider tensor cores (WMMA API) for FP16/BF16 matrix operations on compute 7.0+
- Use `const` and `const&` where appropriate to help compiler optimization

### Error Handling

CUDA errors: `CUDA_CHECK(call)` macro. Throw `std::runtime_error`/`CudaError` for unrecoverable errors. Return `std::expected` for recoverable failures. Benchmark failures: record in results, exit non-zero. Never swallow errors silently.

### Benchmark Interface

Every benchmark implements: `metadata()`, `is_supported(device)`, `setup(ctx, params)`, `run_warmup(ctx, params)`, `run_measure(ctx, params)` → samples + metrics, `teardown(ctx)`.

### GPU Feature Detection

Use `GpuInfo` methods to detect GPU capabilities for feature gating:
- `gpu.is_compute_capability_at_least(major, minor)` - check CC version
- `gpu.supports_warp_shuffle()` - CC 3.0+
- `gpu.supports_tensor_cores()` - CC 7.0+
- `gpu.supports_fp16_tensor_cores()` - CC 7.0+
- `gpu.supports_bf16_tensor_cores()` - CC 8.0+

### Testing

Google Test. Test files: `test_*.cpp`. Run single test: `ctest --test-dir build -R <test_name> -V`. Tests require `-DPERFCLI_ENABLE_TESTS=ON`. Use `TEST(SuiteName, TestName)` macros and `EXPECT_*`/`ASSERT_*` assertions.

### Performance Considerations

Always warm up (amortize JIT/cache). Gather multiple samples; compute robust stats (median, p95, trimmed mean). Ensure workloads large enough to avoid measuring overhead.

### Linting/Formatting

No formal linting configured. Follow the style guidelines above. Ensure code compiles with `-Wall -Wextra` for warnings.

### Adding New Benchmarks

Create `src/benchmarks/<Name>.cu` and `include/perfcli/benchmarks/<Name>.hpp`. Implement interface, register in `src/core/Registry.cpp`.

### Portability Note

CMakeLists.txt contains hardcoded CUDA 13.1 include path; should use `find_package(CUDA)` for portability across CUDA installations.

### Dependencies (via FetchContent)

CLI11, nlohmann/json, fmt. Keep versions pinned.

## Definition of Done

- Compiles without warnings
- Tests pass
- Benchmarks produce plausible metrics
- JSON output parseable and complete
- Documentation updated (if user-facing changes)
