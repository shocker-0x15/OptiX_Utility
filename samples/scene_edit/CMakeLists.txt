set(TARGET_NAME "scene_edit")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

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

nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819\""
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
    ${COMMON_GUI_SOURCES}
    ${TEXTURE_SOURCES}
    ${SOURCES}
    ${OPTIX_KERNELS}
)
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "${TARGET_NAME}"
    FOLDER "samples/applications")
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "../../ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/assimp/include"
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
    assimp::assimp
)
