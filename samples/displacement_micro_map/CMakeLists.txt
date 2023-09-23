set(TARGET_NAME "displacement_micro_map")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp
)

file(
    GLOB CUDA_KERNELS
    "../common/micro_map/micro_map_kernels.cu"
    "../common/micro_map/dmm_kernels.cu"
)

file(
    GLOB DMM_GENERATOR_SOURCES
    "../common/micro_map/micro_map_generator_private.h"
    "../common/micro_map/micro_map_generator.cpp"
    "../common/micro_map/micro_map_generator_misc.cu"
    "../common/micro_map/dmm_generator.h"
    "../common/micro_map/dmm_generator_private.h"
    "../common/micro_map/dmm_generator.cpp"
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
    "../common/micro_map/micro_map_generator_private.h"
    "../common/micro_map/dmm_generator.h"
    "../common/micro_map/dmm_generator_private.h"
    "${TARGET_NAME}_shared.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
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
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    # "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    # "$<$<CONFIG:Debug>:-G>"
    # "$<$<CONFIG:Debug>:-O0>"
    "--gpu-architecture=compute_52"
    "-std=${CPP_VER_CUDA}"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

set_common_sources(TRUE)

# TinyGLTF
file(
    GLOB_RECURSE BARY_FILE_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/Displacement-MicroMap-BaryFile/src/*"
)

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$")
source_group(
    "essentials/dmm_generator" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/samples/common/micro_map/(dmm|micro_map)_.*\.(h|hpp|c|cpp|cu)$")

# non-essentials
source_group(
    "non-essentials/ext/tinygltf" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/tinygltf/tiny_gltf.*\.(h|cc)$")
source_group(
    "non-essentials/ext/bary" REGULAR_EXPRESSION
    "${CMAKE_SOURCE_DIR}/ext/Displacement-MicroMap-BaryFile/src/.*\.(cpp)$")

list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${UTIL_SOURCES}
    ${COMMON_SOURCES}
    ${COMMON_GUI_SOURCES}
    ${TEXTURE_SOURCES}
    ${OBJ_LOADER_SOURCES}
    ${TINY_GLTF_SOURCES}
    ${BARY_FILE_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
    ${DMM_GENERATOR_SOURCES}
)
set_target_properties(
    "${TARGET_NAME}" PROPERTIES
    CXX_EXTENSIONS OFF
    PROJECT_LABEL "14.${TARGET_NAME}"
    FOLDER "samples")
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "USE_CUBD_LIB"
    "TINYGLTF_NO_STB_IMAGE"
    "TINYGLTF_NO_STB_IMAGE_WRITE"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "../../ext/gl3w/include"
    "../../ext/glfw/include"
    "../../ext/imgui"
    "../../ext/Displacement-MicroMap-BaryFile/include"
    "../../ext/tinygltf"
)
target_link_directories(
    "${TARGET_NAME}" PRIVATE
    "../../ext/glfw/lib"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    libcubd_static
    opengl32
    glfw3
)
