set(TARGET_NAME "optixu_tests")

set(TEST_UTIL_SOURCES ${UTIL_SOURCES})
list(
    FILTER TEST_UTIL_SOURCES EXCLUDE REGEX
    "${CMAKE_SOURCE_DIR}/optix_util.cpp$")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

file(
    GLOB OPTIX_KERNELS
    "kernels_0.cu"
    "kernels_1.cu"
)

file(
    GLOB GPU_KERNEL_DEPENDENCIES
    "../../cuda_util.h"
    "../../optix_util.h"
    "../../optixu_on_cudau.h"
    "../samples/common/common.h"
    "shared.h"
)

nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "$<$<CONFIG:Debug>:-O0>"
    "--gpu-architecture=compute_52"
    "-std=${CPP_VER_CUDA}"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

source_group(
    "essentials/utils" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/[^/]*$")

source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$")

add_executable(
    "${TARGET_NAME}"
    ${TEST_UTIL_SOURCES}
    ${SOURCES}
    ${OPTIX_KERNELS}
)
if(${CPP_VER} STREQUAL "c++17")
    target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_17)
elseif(${CPP_VER} STREQUAL "c++20")
    target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
else()
    message(FATAL_ERROR "Invalid C++ standard.")
endif()
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "${TARGET_NAME}"
    FOLDER "tests")
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "CUDA_UTIL_DONT_USE_GL_INTEROP"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    gtest
)
