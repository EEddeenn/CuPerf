# CUDA Performance Test CLI Tool — Software Architecture Plan

This document describes a detailed, implementation-ready architecture for a **CUDA performance test program** built as a **CLI tool** using **modern CUDA** (CUDA 13.1 by default) and **modern C++** (C++23 by default). It is written to be "coding-agent friendly": clear components, responsibilities, file layout, interfaces, and step-by-step milestones.

> **Status**: ✅ Fully implemented, tested, and optimized
> Last updated: 2026-01-12

---

## 1) Goals and non-goals

### Goals
- Provide a **CLI microbenchmark tool**(named Kuda) to characterize GPU performance:
  - Device memory bandwidth (VRAM/HBM)
  - H2D/D2H/D2D copies, pinned vs pageable host memory
  - Kernel launch overhead
  - Compute throughput (FP32/FP16/BF16/INT8 as applicable)
  - Reduction throughput/latency
  - Optional: simple GEMM-like diagnostic kernel (phase 2)
  - Optional: multi-GPU peer copies (phase 2)
- Use **modern CUDA Runtime API** by default (optional Driver API later).
- Produce **reproducible measurements**:
  - Warmup runs
  - Controlled iteration/sample counts
  - Robust stats (median/p95, trimmed mean)
  - Clear synchronization discipline
  - Optional telemetry (NVML) for temperature/power/clocks
- Be **extensible**:
  - Benchmark registry and a clean benchmark interface
  - Add a new benchmark by adding one class + one registry line
- Provide structured output:
  - Human-readable console summary
  - Machine-readable **JSON** (canonical) + optional CSV

### Non-goals (initially out of scope)
- Replacing Nsight/CUPTI profiling. (We may add optional hooks later.)
- Competing with cuBLAS/cuDNN. GEMM-like benchmarks are diagnostic only.

---

## 2) High-level architecture

### Core subsystems
1. **CLI Frontend**
   - Parses args
   - Loads benchmark registry
   - Builds a `RunPlan` (benchmarks, configs, devices)
   - Executes and writes outputs

2. **Benchmark Harness**
   - Unified lifecycle:
     - setup → warmup → measure → teardown
   - Timing, statistics, outlier handling
   - Parameter sweep engine (sizes, dtypes, stream counts, etc.)

3. **Device & Runtime Layer**
   - Device discovery and selection
   - Stream/event management
   - Memory allocators (device + pinned host)
   - Optional NVML telemetry
   - Optional peer access management for multi-GPU

4. **Benchmark Implementations**
   - Each benchmark follows a common interface:
     - metadata + parameters
     - setup/teardown
     - warmup + measure
     - returns samples + derived metrics

5. **Results & Reporting**
   - Data model includes system info, config, benchmark results
   - JSON writer + console renderer; optional CSV

---

## 3) Repository / directory structure

A scalable layout:

cuda-perf-cli/
CMakeLists.txt
cmake/
Toolchains.cmake
Options.cmake
include/
perfcli/
cli/
core/
cuda/
benchmarks/
report/
util/
src/
main.cpp
cli/
Args.cpp
Args.hpp
Commands.cpp
core/
Runner.cpp
Runner.hpp
Registry.cpp
Registry.hpp
RunPlan.cpp
RunPlan.hpp
Statistics.cpp
Statistics.hpp
Sweep.cpp
Sweep.hpp
cuda/
Device.cpp
Device.hpp
Stream.cpp
Stream.hpp
EventTimer.cpp
EventTimer.hpp
Memory.cpp
Memory.hpp
Nvml.cpp (optional)
Nvml.hpp
report/
ConsoleReport.cpp
ConsoleReport.hpp
JsonReport.cpp
JsonReport.hpp
CsvReport.cpp (optional)
 benchmarks/
MemcpyBandwidth.cu
MemcpyBandwidth.hpp
DeviceMemBandwidth.cu
DeviceMemBandwidth.hpp
KernelLaunchOverhead.cu
ComputeThroughput.cu
ComputeThroughput.hpp
Reduction.cu
Reduction.hpp
TensorCore.cu
TensorCore.hpp
third_party/
(optional vendored deps or FetchContent)
tests/
test_statistics.cpp
test_sweep.cpp
docs/
benchmark_design.md
cli_usage.md
reproducibility.md


Notes:
- `.cu` files live under `src/benchmarks` (kernels + measured loops).
- Public headers under `include/perfcli/...` define interfaces and data models.

---

## 4) Build system (CMake)

### Requirements
- CMake ≥ ~3.24 (modern CUDA language support)
- C++23
- CUDA toolkit (assume 12.x, but feature-check instead of hardcoding)

### CMake design
- Single executable target: `perfcli`
- CUDA as a first-class language:

   - `project(perfcli LANGUAGES CXX CUDA)`
   - `set(CMAKE_CXX_STANDARD 23)`
   - `set(CMAKE_CUDA_STANDARD 20)`

- Options:
  - `PERFCLI_ENABLE_NVML` (default ON if found)
  - `PERFCLI_ENABLE_CUPTI` (default OFF)
  - `PERFCLI_ENABLE_CSV` (default OFF)
  - `PERFCLI_ENABLE_STATIC` (optional)

- Performance flags policy:
  - Avoid global `--use_fast_math` by default
  - Enable benchmark-specific flags only where meaningful

### Dependency policy (recommended)
Keep deps minimal and modern:
- CLI parsing: `CLI11` (or `cxxopts`)
- Formatting: `fmt` (or `std::format` if reliably supported)
- JSON: `nlohmann/json` (simple and reliable)

Use `FetchContent` with pinned versions for reproducibility.

---

## 5) Key data models

Implement these as plain structs/classes with JSON serialization.

### System & environment
**`SystemInfo`**:
- Host info (optional): hostname, OS, CPU model
- CUDA runtime version, driver version
- GPU list:
  - name, UUID (if available), PCI bus id
  - compute capability
  - SM count
  - memory size
  - clocks/power limits (if NVML enabled)

### Run configuration
**`RunConfig`**:
- Device selector: index or UUID
- Warmup iterations
- Measured iterations
- Sample count controls (min/max runs per case)
- Outlier policy: none / trimmed mean / winsorize
- Stream count
- Async on/off
- Pinned memory on/off
- Sync mode: event-based vs device-sync-based
- Optional clock/power policy (NVML best-effort)

### Benchmark specification
**`BenchmarkSpec`**:
- Name
- Description
- Parameter schema (typed)
- Default params
- Tags (`memory`, `compute`, `latency`, `multi-gpu`)
- Supported dtypes/features

### Result model
**`BenchmarkResult`**:
- Benchmark metadata + chosen params
- Raw samples (e.g., microseconds)
- Derived metrics (GB/s, GFLOP/s, etc.)
- Summary stats (mean, median, p95, stdev)
- Notes/warnings (e.g., throttling detected)

---

## 6) Benchmark interface and registry

### Interface lifecycle
Each benchmark should provide:
- `metadata()` → name, description, parameter definitions
- `is_supported(device)` → validate CC/features
- `setup(ctx, params)` → allocate memory, create streams/events
- `run_warmup(ctx, params)`
- `run_measure(ctx, params)` → return vector of sample timings + counters
- `teardown(ctx)`

### Registry
Use explicit factory registration:
- `BenchmarkRegistry::register_benchmark("memcpy", [](){ return std::make_unique<MemcpyBandwidth>(); })`
- `BenchmarkRegistry::create(name)` returns an instance

Avoid link-time magic; keep registration explicit.

---

## 7) CUDA timing strategy (correctness-critical)

### Use CUDA events for GPU time
- Create events with `cudaEventCreateWithFlags(..., cudaEventDefault)`
- Record start/stop in the same stream
- Use `cudaEventElapsedTime` (ms float) → convert to µs

### Synchronization discipline
Per-iteration timing:
1. Record start event
2. Enqueue workload
3. Record stop event
4. `cudaEventSynchronize(stop)`

Avoid `cudaDeviceSynchronize()` in inner loops unless explicitly measuring it.

### CPU timing (limited use)
Use `std::chrono` only for:
- Host-only overhead
- CLI, file IO
- Optional end-to-end wall clock mode

### Warmup
Always warm up to amortize:
- context creation
- potential JIT
- cache effects/page faults

### Stats and outliers
Store raw samples and compute robust summaries:
- median, p50/p95/p99
- trimmed mean (drop top/bottom 5% default)
- optional drift detection (throttling)

---

## 8) Core execution flow

1. **Startup**
   - Parse args
   - Discover devices
   - Validate device selection

2. **Build RunPlan**
   - Select benchmarks by name/tag
   - Expand parameter sweeps (sizes, dtypes, stream counts)
   - Apply global config overrides

3. **Execute**
   - For each benchmark case:
     - setup
     - warmup
     - measure (collect samples)
     - compute stats + derived metrics
     - teardown
   - Optionally capture telemetry before/after each case

4. **Report**
   - Print console summary
   - Write JSON output to file/stdout
   - Non-zero exit code if failures occur (benchmark error or verification failure)

---

## 9) CLI design (commands + flags)

### Commands
- `perfcli list`
  - list benchmarks and parameters
- `perfcli info`
  - print system/GPU info
- `perfcli run [benchmarks...]`
  - run selected benchmarks

### Common flags
- `--device 0` or `--device-uuid <uuid>`
- `--tag memory|compute|latency`
- `--json out.json` or `--json -` (stdout)
- `--csv out.csv` (optional)
- `--warmup 50`
- `--iters 200`
- `--samples 30` (aggregate runs per case)
- `--pinned on|off`
- `--streams 1|2|4`
- `--sizes 1K,2K,4K,...`
- `--sizes-range 1K:1G:2x`
- `--dtype fp32|fp16|bf16|int8`
- `--verify on|off`
- `--nvml on|off`
- `--clock-lock <sm_mhz,mem_mhz>` (best-effort; requires permission)

### Examples
- `perfcli list`
- `perfcli info --device 0`
- `perfcli run memcpy --sizes-range 1M:8G:2x --pinned on --iters 50 --json -`
- `perfcli run --tag memory --device 0 --json results.json`

---

## 10) Benchmarks to implement (phased)

### Phase 1 (high value, lower complexity)
1. **MemcpyBandwidth**
   - H2D, D2H, D2D
   - pageable vs pinned
   - sync vs async (`cudaMemcpy` vs `cudaMemcpyAsync`)
   - metrics: GB/s, latency

2. **DeviceMemBandwidth**
   - simple kernels:
     - read-only, write-only, read+write
     - coalesced access
   - sizes: L2-fitting vs larger-than-L2
   - metrics: GB/s

3. **KernelLaunchOverhead**
   - empty kernel
   - per-launch latency (event timing)
   - optional block size sweep
   - metric: µs/launch

4. **ComputeThroughput**
   - FMA-heavy kernel
   - dtypes: fp32, fp16/bf16 if supported
   - metric: GFLOP/s (known ops count)

5. **Reduction**
   - sum reduction (block-level + final)
   - metric: effective GB/s or elements/s, plus latency

### Phase 2 (completed)
6. **TensorCore**
     - WMMA-based GEMM (tensor cores)
     - FP16 matrix multiply (16x16x16 tiles) - CC 7.0+
     - BF16 matrix multiply (16x16x16 tiles) - CC 8.0+
     - INT8 matrix multiply (16x16x16 tiles) - CC 7.2+
     - FP4 matrix multiply (packed storage, uses FP16 tensor ops)
     - metric: TFLOP/s or TOPS
     - requires CC 7.0+ (tensor cores)
     - diagnostic, not a cuBLAS competitor

### Phase 2 (not implemented)
7. **MultiGPU Peer Copy**
    - peer bandwidth/latency across GPUs
    - handle NVLink vs PCIe differences
    - metric: GB/s and µs
    - **Status**: Not yet implemented (future work)

---

## 11) Memory management and allocators

Implement small abstractions to avoid repeated boilerplate.

### Components
- `DeviceBuffer<T>`: `cudaMalloc` / `cudaFree`
- `HostBuffer<T>`:
  - pageable: `new[]` or `std::vector`
  - pinned: `cudaMallocHost` / `cudaFreeHost`
- `BufferView<T>`: pointer + length passed around

### Alignment & initialization
- helpers for:
  - aligned sizes (e.g., 256B)
  - fill patterns for verification
- `cudaMemsetAsync` for device init where appropriate

---

## 12) Telemetry (optional but valuable)

### NVML integration (optional)
- capture:
  - temperature
  - power draw
  - SM/mem clocks
  - utilization (if desired)
- take snapshots before/after each benchmark case
- warn on:
  - thermal throttling signs (clock drop)
  - high temperatures
  - power-limit behavior

Use an interface abstraction:
- `ITelemetryProvider::snapshot() -> TelemetrySnapshot`

Gracefully degrade if NVML isn’t found.

---

## 13) Error handling policy

### CUDA errors
Centralize error handling:
- `CUDA_CHECK(call)` throws a `CudaError` with file/line and `cudaGetErrorString`

After each kernel launch:
- `CUDA_CHECK(cudaGetLastError())`
- optionally (debug mode) `CUDA_CHECK(cudaDeviceSynchronize())`

### Benchmark failures
Benchmarks should return structured failures or throw; the Runner:
- captures error
- records failure in results
- exits non-zero

---

## 14) Verification mode (correctness)

Add `--verify on` to validate outputs:
- memcpy: compare host buffers
- compute: compare against CPU reference for small sizes (or checksum)
- reduction: compare sums

Verification is optional because it can impact performance.

---

## 15) Parameter sweeps

Implement a generic sweep engine:
- Input: parameter schema + user overrides
- Output: list of `CaseConfig`

Support:
- `--sizes-range 4K:1G:2x`
- `--block 128,256,512`
- `--streams 1,2,4`

Implement once; reuse everywhere.

---

## 16) Reporting formats

### Console output
- Per-case summary table:
  - size, dtype, streams
  - median, p95
  - derived metric (GB/s, GFLOP/s)
- End-of-run info:
  - GPU name, driver/runtime versions

### JSON output (canonical)
- includes:
  - system info
  - run config
  - benchmark results + summary stats
  - optional raw samples
- provide:
  - `--json-samples on|off` to control file size

### CSV output (optional)
- flattened rows
- one row per benchmark case:
  - config columns + summary stats

---

## 17) Testing strategy

### Unit tests (CPU-side)
- `Statistics`:
  - percentile correctness
  - trimmed mean behavior
- `Sweep`:
  - expansion correctness
- CLI parsing “golden tests” if feasible

### Smoke tests (GPU, optional)
- `perfcli selftest`
  - tiny kernel + tiny memcpy
  - basic CUDA sanity check

Treat GPU tests as optional in CI.

---

## 18) Performance pitfalls checklist

- Don’t include allocation time in measurements unless explicitly testing alloc performance.
- Don’t create streams/events in inner loops.
- Don’t rely on a single run; gather multiple samples.
- Ensure bandwidth workloads are large enough; otherwise you measure overhead.
- For memcpy benchmarks:
  - measure pageable vs pinned
  - measure async copies for pinned to observe peak path
- For compute:
  - enough work per measurement to avoid launch overhead dominance
- Record device properties and note constraints (power cap, thermals).

---

## 19) Implementation plan (step-by-step milestones)

### Milestone A — Skeleton + device info
1. Set up CMake project (C++20 + CUDA).
2. Add deps (CLI11 + nlohmann/json + fmt).
3. Implement `perfcli info`:
   - enumerate devices
   - print properties
   - optional JSON to stdout (`--json -`)

**Acceptance:** builds and runs; `info` prints GPU name and compute capability.

### Milestone B — Harness + registry
4. Implement core types: `RunConfig`, `BenchmarkSpec`, `BenchmarkResult`, `SystemInfo`.
5. Implement `Benchmark` interface + `BenchmarkRegistry`.
6. Implement `Runner`:
   - device selection
   - run lifecycle
   - results aggregation + JSON emission

**Acceptance:** `perfcli list` shows registered benchmarks; `perfcli run` executes a dummy benchmark.

### Milestone C — Timing + statistics
7. Implement `EventTimer` wrapper and `Statistics`.
8. Add trimming + percentile summaries.

**Acceptance:** results include median/p95; event timing stable across runs.

### Milestone D — Phase 1 benchmarks
9. Implement `MemcpyBandwidth`.
10. Implement `KernelLaunchOverhead`.
11. Implement `DeviceMemBandwidth`.
12. Implement `ComputeThroughput`.
13. Implement `Reduction`.

**Acceptance:** benchmarks run across sweeps and produce plausible metrics.

### Milestone E — Reporting polish
14. Console renderer with tables.
15. Finalize JSON schema + `--json-samples`.
16. Add optional CSV.

**Acceptance:** clean output; JSON stable and parseable.

### Milestone F — Tensor core support
17. Implement `TensorCore` benchmark with WMMA API.
   - FP16 tensor cores (16x16x16)
   - INT8 tensor cores (16x16x16)
   - Configurable GEMM dimensions (--m, --n, --k)

**Acceptance:** `tensor_core` benchmark runs on GPUs with tensor cores and reports TFLOPS/TOPS.

### Milestone G — Optional telemetry
18. Add NVML behind interface; `--nvml` and snapshots per case.

**Acceptance:** works when NVML is available; gracefully disables otherwise.

---

## 20) Definition of done
- ✅ `perfcli list/info/run` functional
- ✅ All 6 core benchmarks implemented (Memcpy, Launch, Mem BW, Compute, Reduction, TensorCore)
- ✅ JSON output includes system info, per-case config, and summary stats
- ✅ Warmup separated from measurement
- ✅ Errors are actionable and structured
- ✅ Adding a new benchmark requires only:
    - a new benchmark class implementing the interface
    - one registry registration line
- ✅ C++23 optimizations (std::span, [[nodiscard]], constexpr, string_view)
- ✅ CUDA kernel optimizations (warp shuffle, unroll, launch_bounds, noinline)
- ✅ Tensor core WMMA API support (FP16, BF16, INT8, FP4)
- ✅ Compute benchmark FP4 support (packed storage)
- Documentation: README.md + AGENTS.md + DESIGN.md

---
