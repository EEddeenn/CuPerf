cmake_minimum_required(VERSION 3.24)
project(perfcli LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;120")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(cmake/Options.cmake)

include(FetchContent)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
)
FetchContent_MakeAvailable(fmt)

if(PERFCLI_ENABLE_NVML)
  message(STATUS "NVML support requested but not implemented yet")
  set(PERFCLI_ENABLE_NVML OFF)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  /usr/local/cuda-13.1/targets/x86_64-linux/include
)

file(GLOB_RECURSE PERFCLI_CXX_SOURCES
  "src/cli/*.cpp"
  "src/core/*.cpp"
  "src/cuda/*.cpp"
)

file(GLOB_RECURSE PERFCLI_CUDA_SOURCES
  "src/benchmarks/*.cpp"
  "src/benchmarks/*.cu"
)

file(GLOB_RECURSE PERFCLI_HEADERS
  "include/**/*.hpp"
  "include/**/*.h"
)

add_executable(perfcli
  src/main.cpp
  ${PERFCLI_CXX_SOURCES}
  ${PERFCLI_CUDA_SOURCES}
  ${PERFCLI_HEADERS}
)

target_link_libraries(perfcli
  PRIVATE
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    fmt::fmt
)

if(PERFCLI_ENABLE_NVML)
  target_compile_definitions(perfcli PRIVATE PERFCLI_WITH_NVML)
endif()

if(PERFCLI_ENABLE_CSV)
  target_compile_definitions(perfcli PRIVATE PERFCLI_ENABLE_CSV)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(perfcli PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
  target_compile_options(perfcli PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(perfcli PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g -O0>)
  target_compile_options(perfcli PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>)
endif()

enable_testing()

add_subdirectory(tests)
