set(TARGET_NAME "opacity_micro_map")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp
)

file(
    GLOB CUDA_KERNELS
    "../common/omm_kernels.cu"
)

file(
    GLOB OMM_GENERATOR_SOURCES
    "../common/omm_generator.h"
    "../common/omm_generator_private.h"
    "../common/omm_generator.cpp"
    "../common/omm_generator_misc.cu"
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
    "../common/omm_generator.h"
    "${TARGET_NAME}_shared.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_52"
    "-std=c++17"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "$<$<CONFIG:Debug>:-O0>"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$")
source_group(
    "essentials/omm_generator" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/samples/common/omm_.*\.(h|hpp|c|cpp|cu)$")

list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${UTIL_SOURCES}
    ${COMMON_SOURCES}
    ${TEXTURE_SOURCES}
    ${OBJ_LOADER_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
    ${OMM_GENERATOR_SOURCES}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "13.${TARGET_NAME}"
    FOLDER "samples")
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "CUDA_UTIL_DONT_USE_GL_INTEROP"
    "USE_CUBD_LIB"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${OPTIX_INCLUDE_DIR}"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    libcubd_static
)
