# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: lswang@openailab.com
#

# Check CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 3.13 FATAL_ERROR)

# Disable inplace builds to prevent source tree corruption.
IF (" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    MESSAGE (FATAL_ERROR "Tengine Fatal: Building inplace are not allowed. You should create a separate directory for Building.")
ENDIF ()

# set cmake_install_prefix path
IF (NOT DEFINED CMAKE_INSTALL_PREFIX)
    SET (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Tengine Installation Directory")
ENDIF()

# Enable the languages which in use
ENABLE_LANGUAGE (C CXX)


# Module options
OPTION (TENGINE_BUILD_BENCHMARK             "Build benchmark"                           ON)
OPTION (TENGINE_BUILD_EXAMPLES              "Build examples"                            ON)
OPTION (TENGINE_BUILD_DEMO                  "Build demos"                               OFF)
OPTION (TENGINE_BUILD_TESTS                 "Build tests"                               OFF)
OPTION (TENGINE_BUILD_CPP_API               "Build C++ API"                             OFF)

# Tools options
# Tools
OPTION (TENGINE_BUILD_CONVERT_TOOL          "Build convert tool"                        OFF)
OPTION (TENGINE_BUILD_QUANT_TOOL            "Build quantization tool"                   OFF)

# Multithreading option
OPTION (TENGINE_OPENMP                      "Build with OpenMP support"                 ON)

# High performance compute library standalone options
OPTION (TENGINE_ARCH_X86_AVX                "Build AVX2 for x86"                        ON)
OPTION (TENGINE_ARCH_ARM_82                 "Build ARM v8.2 for ARM platform"           OFF)

# Standalone HCL options
OPTION (TENGINE_STANDALONE_HCL              "Build standalone hcl lib"                  OFF)
OPTION (TENGINE_STANDALONE_HCL_AUTO_LOAD    "Auto load standalone hcl lib"              ON)

# Plugin options
OPTION (TENGINE_ENABLE_ACL                  "With Arm Compute Library support"          OFF)
OPTION (TENGINE_ENABLE_CUDA                 "With nVIDIA CUDA support"                  OFF)
OPTION (TENGINE_ENABLE_OPENCL               "With Khronos OpenCL support"               OFF)
OPTION (TENGINE_ENABLE_OPENDLA              "With Khronos OpenDLA support"              OFF)
OPTION (TENGINE_ENABLE_TENSORRT             "With nVIDIA TensorRT support"              OFF)
OPTION (TENGINE_ENABLE_TIM_VX               "With VeriSilicon TIM-VX support"           OFF)
OPTION (TENGINE_ENABLE_TORCH                "With libTorch support"                     OFF)
OPTION (TENGINE_ENABLE_NNIE                 "With HiSilicon NNIE support"               OFF)
OPTION (TENGINE_ENABLE_VULKAN               "With Khronos Vulkan GPU compute support"   OFF)

# Debug options
OPTION (TENGINE_DEBUG_DATA                  "Extract feature map for each layer"        OFF)
OPTION (TENGINE_DEBUG_TIME                  "Print execution time for each layer"       OFF)
OPTION (TENGINE_DEBUG_MEM_STAT              "Print memory status for library"           OFF)
OPTION (TENGINE_ENABLE_ALL_SYMBOL           "All symbol visible"                        OFF)

# Experimental optinos
OPTION (TENGINE_ENABLE_MODEL_CACHE          "NPU kernel cache file option"              OFF)

# Online report
OPTION (TENGINE_ONLINE_REPORT               "online report"                             OFF)

OPTION (TENGINE_RV64_RVV_C906               "build for c906"                            OFF)
OPTION (TENGINE_COVERAGE                    "build with coverage info"                  OFF)

IF (CMAKE_TOOLCHAIN_FILE)
    SET (LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")

    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    GET_FILENAME_COMPONENT (CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    FIND_FILE (CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    MESSAGE (STATUS "Using CMake tool chain file ${CMAKE_TOOLCHAIN_FILE}")
ENDIF()

IF (NOT CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
ENDIF()

# Do check list
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/check.cmake")
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda.cmake")
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch.cmake")
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/registry.cmake")
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utility.cmake")

# Project name
PROJECT (tengine-lite)

# enable assembly language
ENABLE_LANGUAGE (ASM)

# Enable cmake ctest module
ENABLE_TESTING ()

# Set cmake default folder name
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET_PROPERTY(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

# Main source files
ADD_SUBDIRECTORY (source)

# Internal modules, not for open source part
INCLUDE (internal/extension.cmake OPTIONAL)

IF (TENGINE_BUILD_BENCHMARK)
    ADD_SUBDIRECTORY(benchmark)
ENDIF()

IF (TENGINE_BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()

IF (TENGINE_BUILD_DEMO)
    ADD_SUBDIRECTORY(demos)
ENDIF()

IF (TENGINE_BUILD_TESTS)
    ADD_SUBDIRECTORY(tests)
ENDIF()

IF (TENGINE_BUILD_QUANT_TOOL)
    ADD_SUBDIRECTORY(tools/quantize)
ENDIF()

IF (TENGINE_BUILD_CONVERT_TOOL)
    ADD_SUBDIRECTORY(tools/convert_tool)
ENDIF()

# Report summary
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/summary.cmake")
