# Agent Guidelines for Kuda (CUDA Performance CLI Tool)

This file provides coding agents with essential information to work effectively in this repository.

## Build System

This project uses CMake with C++20 and modern CUDA (targeting CUDA 13.1/12.x).

### Essential Commands

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel $(nproc)

# Clean
rm -rf build build-debug

# Run the CLI
./build/perfcli --help
./build/perfcli info
./build/perfcli list
./build/perfcli run memcpy --sizes-range 1M:8G:2x --json results.json

# Run single test (when tests exist)
ctest --test-dir build -R <test_name> -V
```

### CMake Options

- `PERFCLI_ENABLE_NVML` (default: auto-detect)
- `PERFCLI_ENABLE_CUPTI` (default: OFF)
- `PERFCLI_ENABLE_CSV` (default: OFF)
- `PERFCLI_ENABLE_STATIC` (optional)

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

### Type System

- Use strong types over primitives where semantics matter (e.g., `DeviceIndex` instead of `int`)
- Prefer `std::chrono` for timing instead of raw integers
- Use `std::optional` for nullable values, not raw pointers
- Use `std::expected` (C++23) or error codes for recoverable errors
- Use `std::span` for array views (C++20)
- Avoid `auto` where the type is not obvious

### Memory Management

- Use RAII for all resources
- `DeviceBuffer<T>` wrapper for `cudaMalloc`/`cudaFree`
- `HostBuffer<T>` wrapper for host memory (pageable or pinned)
- Never use raw `new`/`delete`; prefer `std::make_unique`/`std::make_shared`
- Follow rule of five/five-zero for resource-managing classes

### CUDA-Specific Guidelines

- Always check CUDA errors: `CUDA_CHECK(call)` macro throws `CudaError`
- Use CUDA events for GPU timing, not `cudaDeviceSynchronize()` in inner loops
- Timing pattern: record start → enqueue → record stop → `cudaEventSynchronize(stop)`
- Record `cudaGetLastError()` after kernel launches
- Don't include allocation time in measurements unless explicitly testing alloc performance
- Don't create streams/events in inner loops
- Prefer `cudaMemcpyAsync` with pinned memory for async copies

### Error Handling

- CUDA errors: Use `CUDA_CHECK(call)` macro with file/line info
- Throw exceptions (`std::runtime_error`, `CudaError`) for unrecoverable errors
- Return `std::expected` or error codes for recoverable failures
- Benchmark failures: Record in results, exit non-zero
- Never swallow errors silently

### Benchmark Interface

Every benchmark must implement:
- `metadata()` → BenchmarkSpec (name, description, parameters)
- `is_supported(device)` → validate device capabilities
- `setup(ctx, params)` → allocate resources
- `run_warmup(ctx, params)` → warmup iterations
- `run_measure(ctx, params)` → return sample timings + metrics
- `teardown(ctx)` → cleanup

### Testing

- Use Google Test for unit tests (when added)
- Test filename pattern: `test_*.cpp`
- Run single test: `ctest -R <test_name> -V`
- Smoke test: `perfcli selftest` (to be implemented)

### Performance Considerations

- Always warm up before measuring (amortize context/JIT/cache effects)
- Gather multiple samples; compute robust stats (median, p95, trimmed mean)
- Ensure workloads are large enough to avoid measuring overhead
- Document any trade-offs between accuracy and performance

### Adding New Benchmarks

To add a benchmark:
1. Create `src/benchmarks/<Name>.cu` and `include/perfcli/benchmarks/<Name>.hpp`
2. Implement the benchmark interface
3. Register in `src/core/Registry.cpp`: `BenchmarkRegistry::register_benchmark("<name>", ...)`
4. Add to design docs if parameters or metrics differ from standard patterns

### Dependencies (via FetchContent)

- `CLI11` (or `cxxopts`) for CLI parsing
- `nlohmann/json` for JSON output
- `fmt` (or `std::format`) for formatting

Keep versions pinned for reproducibility.

## Output Format Requirements

- Console: Human-readable tables with per-case summaries
- JSON: Canonical format with system info, run config, and benchmark results
- Optional CSV (flattened rows)

## Definition of Done

- Code compiles without warnings
- Lints pass (when linting is configured)
- Tests pass
- Benchmarks produce plausible metrics
- JSON output is parseable and complete
- Documentation updated (if user-facing changes)
