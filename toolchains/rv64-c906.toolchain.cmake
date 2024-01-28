# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR riscv64)

# riscv64-unknown-linux-gnu DO NOT need to be installed, so make sure riscv64-unknown-linux-gnu-gcc and riscv64-unknown-linux-gnu-g++ can be found in $PATH:
SET (CMAKE_C_COMPILER   "riscv64-unknown-linux-gnu-gcc")
SET (CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# other needed options
IF (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    SET (TENGINE_TOOLCHAIN_ASM_FLAG -march=rv64gcv -mabi=lp64d -mtune=thead-c906 -lc)
ELSE()
    SET (TENGINE_TOOLCHAIN_ASM_FLAG -march=rv64gcv -mabi=lp64d -g -O0 -lc)
ENDIF()

IF (TENGINE_RV64_RVV_C906)
    SET(TENGINE_TOOLCHAIN_ASM_FLAG "-D__FIX_RVV_C906 ${TENGINE_TOOLCHAIN_ASM_FLAG}")
ENDIF()

#SET (TENGINE_TOOLCHAIN_FLAG -march=rv64imafdcvxtheadc -mabi=lp64dv -mtune=c906 -mfp16)
#SET (TENGINE_TOOLCHAIN_FLAG -march=rv64imafdcvxtheadc -mabi=lp64dv -mtune=c910 -mfp16)

# skip OpenMP check as default, for early RISC-V only have single CPU core
SET (TENGINE_FORCE_SKIP_OPENMP ON)

# Note:
#   Early toolchain was not usable enough, so users should adjust TENGINE_TOOLCHAIN_FLAG for a compilation pass
