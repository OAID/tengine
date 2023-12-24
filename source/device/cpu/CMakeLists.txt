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

# unset for generation
UNSET (_DEV_CPU_HEADER_PATH)
UNSET (_CPU_OP_DIR)
UNSET (_CPU_OP_LIST)
UNSET (_CPU_DEVICE_SOURCE)
UNSET (_CPU_REF_SOURCE)
UNSET (_CPU_HCL_SOURCE)
UNSET (_CPU_CORTEX_A_SOURCE)
UNSET (_CPU_CORTEX_A_V7_SOURCE)
UNSET (_CPU_CORTEX_A_V7_KERNEL)
UNSET (_CPU_CORTEX_A_V8_SOURCE)
UNSET (_CPU_CORTEX_A_V8_KERNEL)
UNSET (_CPU_CORTEX_A_V8_2_SOURCE)
UNSET (_CPU_CORTEX_A_V8_2_KERNEL)
UNSET (_CPU_CORTEX_M_SOURCE)
UNSET (_CPU_CORTEX_M_KERNEL)
UNSET (_CPU_x86_SOURCE)
UNSET (_CPU_x86_KERNEL)
UNSET (_CPU_MIPS_SOURCE)
UNSET (_CPU_MIPS_KERNEL)
UNSET (_CPU_REGISTER_SOURCE)

UNSET (_CPU_COMPILER_DEFINES)
UNSET (_CPU_COMPILER_OPTIONS)



# 0. add header file path
LIST (APPEND _DEV_CPU_HEADER_PATH "${CMAKE_SOURCE_DIR}/source/device")
LIST (APPEND _DEV_CPU_HEADER_PATH "${CMAKE_BINARY_DIR}/source/device/cpu")
LIST (APPEND _DEV_CPU_HEADER_PATH "${CMAKE_SOURCE_DIR}/source/device/cpu")
LIST (APPEND _DEV_CPU_HEADER_PATH "${CMAKE_SOURCE_DIR}/source/operator/prototype")


# 1. add source files
# 1.1 set source root path
SET(_OP_ROOT ${CMAKE_SOURCE_DIR}/source/device/cpu/op)


# 1.2 get all op folder from source dir
FILE (GLOB _CPU_OP_DIR RELATIVE ${_OP_ROOT} ${_OP_ROOT}/*)


# 1.3 check if define allowed operators list
IF ((DEFINED TENGINE_CPU_OPS_LIMIT_FILE) AND (EXISTS ${TENGINE_CPU_OPS_LIMIT_FILE}))
    # 1.3.1 get allowed op options
    INCLUDE ("${TENGINE_CPU_OPS_LIMIT_FILE}")

    # 1.3.2 check if op is enabled
    FOREACH(_DIR ${_CPU_OP_DIR})
        IF (FIND _CPU_OP_LIST ${_OP_NAME} _OP_NAME_FOUND)
            IF ((NOT _OP_NAME_FOUND) AND ${TENGINE_OP_${_OP_NAME}})
                LIST (APPEND _CPU_OP_LIST ${_OP_NAME})
                IF(${TENGINE_VERBOSE})
                    MESSAGE(AUTHOR_WARNING "  add op ${_OP_NAME}")
                ENDIF()
            ENDIF()
        ENDIF()
    ENDFOREACH()
ELSE()
    # 1.3.1 if not set limit file, all op is allowed
    FOREACH(_DIR ${_CPU_OP_DIR})
        LIST (APPEND _CPU_OP_LIST ${_DIR})
        IF(${TENGINE_VERBOSE})
            MESSAGE(AUTHOR_WARNING "  add op ${_DIR}")
        ENDIF()
    ENDFOREACH()
ENDIF()


# 1.4 collect all source files
# 1.4.1 collect source files for cpu device
FILE (GLOB _CPU_DEVICE_SOURCE "${CMAKE_SOURCE_DIR}/source/device/cpu/*.c")

# 1.4.2 collect source files for cpu operators
FOREACH(_OP_NAME ${_CPU_OP_LIST})
    # base op files should be placed at ${_OP_NAME}/
    # base op main file should be name with "${_OP_NAME}_ref.c"
    FILE (GLOB _REF_SOURCE                  "${_OP_ROOT}/${_OP_NAME}/*.c")

    FILE (GLOB _CORTEX_A_SOURCE             "${_OP_ROOT}/${_OP_NAME}/cortex-a/*.c")
    FILE (GLOB _CORTEX_A_V7_SOURCE          "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv7/*.c")
    FILE (GLOB _CORTEX_A_V7_KERNEL          "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv7/*.S")
    FILE (GLOB _CORTEX_A_V8_SOURCE          "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv8/*.c")
    FILE (GLOB _CORTEX_A_V8_KERNEL          "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv8/*.S")
    FILE (GLOB _CORTEX_A_V8_2_SOURCE        "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv8.2/*.c")
    FILE (GLOB _CORTEX_A_V8_2_KERNEL        "${_OP_ROOT}/${_OP_NAME}/cortex-a/armv8.2/*.S")

    FILE (GLOB _CORTEX_M_SOURCE             "${_OP_ROOT}/${_OP_NAME}/cortex-m/*.c")
    FILE (GLOB _CORTEX_M_KERNEL             "${_OP_ROOT}/${_OP_NAME}/cortex-m/*.S")

    FILE (GLOB _x86_SOURCE                  "${_OP_ROOT}/${_OP_NAME}/x86/*.c")
    FILE (GLOB _x86_KERNEL                  "${_OP_ROOT}/${_OP_NAME}/x86/*.S")

    FILE (GLOB _MIPS_SOURCE                 "${_OP_ROOT}/${_OP_NAME}/mips/*.c")
    FILE (GLOB _MIPS_KERNEL                 "${_OP_ROOT}/${_OP_NAME}/mips/*.S")

    FILE (GLOB _RISC_V_SOURCE               "${_OP_ROOT}/${_OP_NAME}/risc-v/*.c")
    FILE (GLOB _RISC_V_KERNEL               "${_OP_ROOT}/${_OP_NAME}/risc-v/*.S")
    FILE (GLOB _RISC_V_LP64DV_SOURCE        "${_OP_ROOT}/${_OP_NAME}/risc-v/lp64dv/*.c")
    FILE (GLOB _RISC_V_LP64DV_KERNEL        "${_OP_ROOT}/${_OP_NAME}/risc-v/lp64dv/*.S")

    LIST (APPEND _CPU_REF_SOURCE            ${_REF_SOURCE})
    LIST (APPEND _CPU_CORTEX_A_SOURCE       ${_CORTEX_A_SOURCE})
    LIST (APPEND _CPU_CORTEX_A_V7_SOURCE    ${_CORTEX_A_V7_SOURCE})
    LIST (APPEND _CPU_CORTEX_A_V7_KERNEL    ${_CORTEX_A_V7_KERNEL})
    LIST (APPEND _CPU_CORTEX_A_V8_SOURCE    ${_CORTEX_A_V8_SOURCE})
    LIST (APPEND _CPU_CORTEX_A_V8_KERNEL    ${_CORTEX_A_V8_KERNEL})
    LIST (APPEND _CPU_CORTEX_A_V8_2_SOURCE  ${_CORTEX_A_V8_2_SOURCE})
    LIST (APPEND _CPU_CORTEX_A_V8_2_KERNEL  ${_CORTEX_A_V8_2_KERNEL})
    LIST (APPEND _CPU_CORTEX_M_SOURCE       ${_CORTEX_M_SOURCE})
    LIST (APPEND _CPU_CORTEX_M_KERNEL       ${_CORTEX_M_KERNEL})
    LIST (APPEND _CPU_x86_SOURCE            ${_x86_SOURCE})
    LIST (APPEND _CPU_x86_KERNEL            ${_x86_KERNEL})
    LIST (APPEND _CPU_MIPS_SOURCE           ${_MIPS_SOURCE})
    LIST (APPEND _CPU_MIPS_KERNEL           ${_MIPS_KERNEL})
    LIST (APPEND _CPU_RISC_V_LP64DV_SOURCE  ${_RISC_V_LP64DV_SOURCE})
    LIST (APPEND _CPU_RISC_V_LP64DV_KERNEL  ${_RISC_V_LP64DV_KERNEL})
ENDFOREACH()


# 1.5 collect all need be registered files
FOREACH(_OP_NAME ${_CPU_OP_LIST})
    # base op files should be placed at ${_OP_NAME}/
    # base op main file should be name with "${_OP_NAME}_ref.c"
    FILE (GLOB _CPU_REF_REGISTER_FILE   "${_OP_ROOT}/${_OP_NAME}/${_OP_NAME}_ref.c")
    FILE (GLOB _CORTEX_A_REGISTER_FILE  "${_OP_ROOT}/${_OP_NAME}/cortex-a/*_hcl_arm.c")
    FILE (GLOB _CORTEX_M_REGISTER_FILE  "${_OP_ROOT}/${_OP_NAME}/cortex-m/*_cmsis.c")
    FILE (GLOB _x86_REGISTER_FILE       "${_OP_ROOT}/${_OP_NAME}/x86/*_hcl_x86.c")
    FILE (GLOB _MIPS_REGISTER_FILE      "${_OP_ROOT}/${_OP_NAME}/mips/*_hcl_mips.c")
    FILE (GLOB _RISC_V_REGISTER_FILE    "${_OP_ROOT}/${_OP_NAME}/risc-v/lp64dv/*_hcl_rv64.c")
    FILE (GLOB _RISC_V_REGISTER_FILE    "${_OP_ROOT}/${_OP_NAME}/risc-v/lp64dv/*_hcl_rv64_tile8.c")

    LIST (APPEND _CPU_REGISTER_SOURCE ${_CPU_REF_REGISTER_FILE})
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        LIST (APPEND _CPU_REGISTER_SOURCE ${_CORTEX_A_REGISTER_FILE})
    ENDIF()
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "Cortex-M")
        LIST (APPEND _CPU_REGISTER_SOURCE ${_CORTEX_M_REGISTER_FILE})
    ENDIF()
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        LIST (APPEND _CPU_REGISTER_SOURCE ${_x86_REGISTER_FILE})
    ENDIF()
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        LIST (APPEND _CPU_REGISTER_SOURCE ${_MIPS_REGISTER_FILE})
    ENDIF()
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
        LIST (APPEND _CPU_REGISTER_SOURCE ${_RISC_V_REGISTER_FILE})
    ENDIF()
ENDFOREACH()


# 1.6 collect all source files to var
# 1.6.1 collect operator for Cortex-A source files
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_SOURCE})
    TENGINE_SOURCE_GROUP ("device/cpu/cortex-a" ${_CPU_CORTEX_A_SOURCE} )

    IF (TENGINE_TARGET_PROCESSOR_32Bit)
        LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V7_SOURCE})
        LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V7_KERNEL})
        TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv7/source" ${_CPU_CORTEX_A_V7_SOURCE})
        TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv7/kernel" ${_CPU_CORTEX_A_V7_KERNEL})
    ENDIF()
    IF (TENGINE_TARGET_PROCESSOR_64Bit)
        LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V8_SOURCE})
        LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V8_KERNEL})
        TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv8/source" ${_CPU_CORTEX_A_V8_SOURCE})
        TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv8/kernel" ${_CPU_CORTEX_A_V8_KERNEL})

        IF (TENGINE_ARCH_ARM_82)
            LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V8_2_SOURCE})
            LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_A_V8_2_KERNEL})
            TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv8.2/source" ${_CPU_CORTEX_A_V8_2_SOURCE})
            TENGINE_SOURCE_GROUP ("device/cpu/cortex-a/armv8.2/kernel" ${_CPU_CORTEX_A_V8_2_KERNEL})
        ENDIF()
    ENDIF()
ENDIF()

# 1.6.2 collect operator for Cortex-M source files
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "Cortex-M")
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_M_SOURCE})
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_CORTEX_M_KERNEL})
    TENGINE_SOURCE_GROUP ("device/cpu/cortex-m/source" ${_CPU_CORTEX_M_SOURCE})
    TENGINE_SOURCE_GROUP ("device/cpu/cortex-m/kernel" ${_CPU_CORTEX_M_KERNEL})
ENDIF()

# 1.6.3 collect operator for x86 source files
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_x86_SOURCE})
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_x86_KERNEL})
    TENGINE_SOURCE_GROUP ("device/cpu/x86/source" ${_CPU_x86_SOURCE})
    TENGINE_SOURCE_GROUP ("device/cpu/x86/kernel" ${_CPU_x86_KERNEL})
ENDIF()

# 1.6.4 collect operator for x86 source files
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_MIPS_SOURCE})
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_MIPS_KERNEL})
    TENGINE_SOURCE_GROUP ("device/cpu/mips/source" ${_CPU_MIPS_SOURCE})
    TENGINE_SOURCE_GROUP ("device/cpu/mips/kernel" ${_CPU_MIPS_KERNEL})
ENDIF()

# 1.6.5 collect operator for RISC-V lp64dv source files
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_RISC_V_LP64DV_SOURCE})
    LIST (APPEND _CPU_HCL_SOURCE ${_CPU_RISC_V_LP64DV_KERNEL})
    TENGINE_SOURCE_GROUP ("device/cpu/risc-v/source" ${_CPU_RISC_V_LP64DV_SOURCE})
    TENGINE_SOURCE_GROUP ("device/cpu/risc-v/kernel" ${_CPU_RISC_V_LP64DV_KERNEL})
ENDIF()


# 7. execute configuration to register operators
GENERATE_REGISTER_HEADER_FILE ("register_" "unregister_" "_op" "${CMAKE_SOURCE_DIR}/source/device/cpu/cpu_ops.h.in" "${CMAKE_BINARY_DIR}/source/device/cpu/cpu_ops.h" "${_CPU_REGISTER_SOURCE}")


# 8. add build optioncds for cpu device
# 8.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        IF (TENGINE_TARGET_PROCESSOR_32Bit)
            IF ((NOT (DEFINED TENGINE_TOOLCHAIN_FLAG)) AND (NOT ANDROID) AND (NOT CMAKE_CROSSCOMPILING) OR (CMAKE_CROSSCOMPILING AND OHOS))
                LIST (APPEND _CPU_COMPILER_OPTIONS "-march=armv7-a")
                LIST (APPEND _CPU_COMPILER_OPTIONS "-mfpu=neon")

                # need check float point when target process is aarch32
                INCLUDE (${PROJECT_SOURCE_DIR}/cmake/environments/check_arm_hard_fp.cmake)
                TENGINE_CHECK_ARM32_HARD_FP(_is_hard_float_point)

                IF(${_is_hard_float_point})
                    LIST (APPEND _CPU_COMPILER_OPTIONS "-mfloat-abi=hard")
                ELSE()
                    LIST (APPEND _CPU_COMPILER_OPTIONS "-mfloat-abi=softfp")
                ENDIF()
            ENDIF()

            IF (NOT ANDROID AND NOT OHOS)
                LIST (APPEND _CPU_COMPILER_OPTIONS "-mfp16-format=ieee")
                #LIST (APPEND _CPU_COMPILER_OPTIONS "-mfpu=neon-fp16")
            ENDIF()
        ENDIF()

        IF (TENGINE_TARGET_PROCESSOR_64BIT AND ${TENGINE_ARCH_ARM_82})
            LIST (APPEND _CPU_COMPILER_OPTIONS "-march=armv8.2-a+fp16")
        ENDIF()
    ENDIF()

    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "X86" AND ${TENGINE_ARCH_X86_AVX})
        LIST (APPEND _CPU_COMPILER_OPTIONS "-mfma")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-mf16c")
    ENDIF()

    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-mabi=64")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-mmsa")

        LIST (APPEND _CPU_LINKER_OPTIONS   "-mabi=64")
        LIST (APPEND _CPU_LINKER_OPTIONS   "-mmsa")
    ENDIF()

    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-march=rv64gcvxthead3")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-mabi=lp64d")
        LIST (APPEND _CPU_COMPILER_OPTIONS "-lc")
    ENDIF()    
ENDIF()

# 8.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
    LIST (APPEND _CPU_COMPILER_DEFINES "NOMINMAX")
    LIST (APPEND _CPU_COMPILER_DEFINES "_SCL_SECURE_NO_WARNINGS")
    LIST (APPEND _CPU_COMPILER_DEFINES "_CRT_SECURE_NO_DEPRECATE")
    LIST (APPEND _CPU_COMPILER_DEFINES "_ENABLE_EXTENDED_ALIGNED_STORAGE")

    LIST (APPEND _CPU_COMPILER_OPTIONS "/MP")

    IF (${TENGINE_TARGET_PROCESSOR} MATCHES "X86" AND ${TENGINE_ARCH_X86_AVX})
        LIST (APPEND _CPU_COMPILER_OPTIONS "/arch:AVX2")
    ENDIF()
ENDIF()


# 9. set all to cmake cache
SET (TENGINE_CPU_HEADER_PATH        ${_DEV_CPU_HEADER_PATH}     CACHE INTERNAL  "Tengine CPU device header files searching path"    FORCE)
SET (TENGINE_CPU_DEVICE_SOURCE      ${_CPU_DEVICE_SOURCE}       CACHE INTERNAL  "Tengine CPU device main source files"              FORCE)
SET (TENGINE_CPU_REF_SOURCE         ${_CPU_REF_SOURCE}          CACHE INTERNAL  "Tengine CPU device reference op file"              FORCE)
SET (TENGINE_CPU_HCL_SOURCE         ${_CPU_HCL_SOURCE}          CACHE INTERNAL  "Tengine CPU device high performance op file"       FORCE)
SET (TENGINE_CPU_COMPILER_DEFINES   ${_CPU_COMPILER_DEFINES}    CACHE INTERNAL  "Tengine CPU about compiler defines"                FORCE)
SET (TENGINE_CPU_COMPILER_OPTIONS   ${_CPU_COMPILER_OPTIONS}    CACHE INTERNAL  "Tengine CPU about compiler options"                FORCE)
SET (TENGINE_CPU_LINKER_OPTIONS     ${_CPU_LINKER_OPTIONS}      CACHE INTERNAL  "Tengine CPU about linker options"                  FORCE)
