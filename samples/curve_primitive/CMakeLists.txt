set(TARGET_NAME "curve_primitive")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp
    "../common/curve_evaluator.h"
)

file(
    GLOB OPTIX_KERNELS
    "optix_kernels.cu"
)

file(
    GLOB GPU_KERNEL_DEPENDENCIES
    "../../cuda_util.h"
    "../../optix_util.h"
    "../../optixu_on_cudau.h"
    "../common/common.h"
    "../common/curve_evaluator.h"
    "${TARGET_NAME}_shared.h"
)

nvcuda_compile_ptx(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-lineinfo>"
    "--gpu-architecture=compute_52"
    "-std=c++17"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

set_common_sources(FALSE)

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$|${CMAKE_SOURCE_DIR}/samples/common/curve_evaluator.h")

add_executable(
    "${TARGET_NAME}"
    ${UTIL_SOURCES}
    ${COMMON_SOURCES}
    ${OBJ_LOADER_SOURCES}
    ${SOURCES}
    ${OPTIX_KERNELS}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "06.${TARGET_NAME}"
    FOLDER "samples")
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
)
