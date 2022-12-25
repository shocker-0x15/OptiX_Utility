set(TARGET_NAME "as_update")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

file(
    GLOB CUDA_KERNELS
    "deform.cu"
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
    "${TARGET_NAME}_shared.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_52"
    "-std=${CPP_VER_CUDA}"
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
    "-std=${CPP_VER_CUDA}"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

set_common_sources(TRUE)

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$")

add_executable(
    "${TARGET_NAME}"
    ${UTIL_SOURCES}
    ${COMMON_SOURCES}
    ${COMMON_SHADERS}
    ${OBJ_LOADER_SOURCES}
    ${GL_SOURCES}
    ${IMGUI_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
)
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "04.${TARGET_NAME}"
    FOLDER "samples")
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "../../ext/gl3w/include"
    "../../ext/glfw/include"
    "../../ext/imgui"
)
target_link_directories(
    "${TARGET_NAME}" PRIVATE
    "../../ext/glfw/lib"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    opengl32
    glfw3
)
