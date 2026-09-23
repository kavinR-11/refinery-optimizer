cmake_minimum_required(VERSION 3.20)
project(sih26119_solver VERSION 0.1.0 LANGUAGES C CXX)

# System and C++ Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA Option (Default: OFF)
option(ENABLE_CUDA "Enable CUDA acceleration with sm_89" OFF)
if(ENABLE_CUDA)
    message(STATUS "CUDA acceleration enabled.")
    set(CMAKE_CUDA_HOST_COMPILER "g++-13")
    set(CMAKE_CUDA_ARCHITECTURES "89")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(SIH_CUDA_ENABLED)
endif()

# Compiler Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter -O3)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# Python & Pybind11
find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 REQUIRED)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

# Core Library Sources
set(SIH_CORE_SOURCES
    core/src/model/sparse_matrix.cpp
    core/src/model/problem.cpp
    core/src/model/solution.cpp
    core/src/model/options.cpp
    core/src/io/mps_io.cpp
    core/src/checker/checker.cpp
    core/src/telemetry/telemetry.cpp
    core/src/utils/affinity.cpp
    core/src/scaling/scaler.cpp
    core/src/factorization/sparse_lu.cpp
)

add_library(sih_core STATIC ${SIH_CORE_SOURCES})
target_link_libraries(sih_core PUBLIC OpenMP::OpenMP_CXX pthread)
target_include_directories(sih_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

# Python Extension Module (_core)
pybind11_add_module(_core core/bindings/pybind_module.cpp)
target_link_libraries(_core PRIVATE sih_core)
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/py/sih_solver"
)

# Enable Testing
enable_testing()
add_subdirectory(tests)
