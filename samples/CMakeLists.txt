# ----------------------------------------------------------------
# JP: プロジェクト間で共通のソースコードのリスト作成。

file(
    GLOB UTIL_SOURCES
    "${CMAKE_SOURCE_DIR}/cuda_util.h"
    "${CMAKE_SOURCE_DIR}/cuda_util.cpp"
    "${CMAKE_SOURCE_DIR}/optix_util.h"
    "${CMAKE_SOURCE_DIR}/optix_util_private.h"
    "${CMAKE_SOURCE_DIR}/optix_util.cpp"
)

file(
    GLOB_RECURSE COMMON_SOURCES
    "common/common.h"
    "common/common.cpp"
    "${CMAKE_SOURCE_DIR}/ext/stb_image_write.h"
)

file(
    GLOB_RECURSE COMMON_SHADERS
    "${CMAKE_SOURCE_DIR}/samples/common/shaders/*"
)

file(
    GLOB TEXTURE_SOURCES
    "common/dds_loader.h"
    "common/dds_loader.cpp"
    "${CMAKE_SOURCE_DIR}/ext/stb_image.h"
)

# obj loader
file(
    GLOB OBJ_LOADER_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/tinyobjloader/tiny_obj_loader.h"
    "${CMAKE_SOURCE_DIR}/ext/tinyobjloader/tiny_obj_loader.cc"
    "common/obj_loader.h"
    "common/obj_loader.cpp"
)

# OpenGL
file(
    GLOB_RECURSE GL3W_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/gl3w/*"
)
file(
    GLOB_RECURSE GLFW_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/glfw/*"
)
file(
    GLOB GLU_SOURCES
    "common/gl_util.h"
    "common/gl_util.cpp"
)
set(
    GL_SOURCES
    ${GL3W_SOURCES}
    ${GLFW_SOURCES}
    ${GLU_SOURCES}
)
list(
    FILTER GL_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|c|cc|cpp)$")

# ImGui
file(
    GLOB_RECURSE IMGUI_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/imgui/*"
)
list(
    FILTER IMGUI_SOURCES EXCLUDE REGEX
    "${CMAKE_SOURCE_DIR}/ext/imgui/imgui_demo\.cpp$")
list(
    FILTER IMGUI_SOURCES EXCLUDE REGEX
    "${CMAKE_SOURCE_DIR}/ext/imgui/(backends|misc|examples)/.*")
file(
    GLOB IMGUI_BACKEND_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/imgui/backends/imgui_impl_glfw*"
    "${CMAKE_SOURCE_DIR}/ext/imgui/backends/imgui_impl_opengl3*"
)
file(
    GLOB IMGUI_FILE_DIALOG_SOURCES
    "common/imgui_file_dialog.h"
    "common/imgui_file_dialog.cpp"
)
set(
    IMGUI_SOURCES
    ${IMGUI_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
    ${IMGUI_FILE_DIALOG_SOURCES}
)
list(
    FILTER IMGUI_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|c|cc|cpp)$")

# END:JP: プロジェクト間で共通のソースコードのリスト作成。
# ----------------------------------------------------------------



# Called from the sub directories
function(set_common_sources HAS_GUI)
    if(${HAS_GUI})
        copy_files(
            SOURCES ${COMMON_SHADERS}
            TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/shaders"
            GENERATED_FILES COPIED_COMMON_SHADERS
        )
    endif()

    # non-essentials
    source_group(
        "non-essentials/ext" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/[^/]*\.(h|hpp|c|cc|cpp)$")
    source_group(
        "non-essentials/ext/tinyobjloader" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/tinyobjloader/[^/]*\.(h|cc)$")
    source_group(
        "non-essentials/ext/imgui/core" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/imgui/[^/]*\.(h|cpp)$")
    source_group(
        "non-essentials/ext/imgui" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/imgui/backends/.*\.(h|cpp)$")
    source_group(
        "non-essentials/ext/gl3w" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/gl3w/.*$")
    source_group(
        "non-essentials/ext/glfw" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/ext/glfw/.*\.h$")
    source_group(
        "non-essentials/shaders" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/samples/common/shaders/.*")
    source_group(
        "non-essentials" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/samples/common/[^/]*\.(h|hpp|c|cpp)$")

    # essentials
    source_group(
        "essentials/utils" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/[^/]*$")
endfunction()



add_library(fakelib INTERFACE)
target_compile_definitions(
    fakelib INTERFACE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    fakelib INTERFACE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
)

add_subdirectory(single_gas)
add_subdirectory(single_level_instancing)
add_subdirectory(texture)
add_subdirectory(material_sets)
add_subdirectory(as_update)
add_subdirectory(custom_primitive)
add_subdirectory(curve_primitive)
add_subdirectory(multi_level_instancing)
add_subdirectory(deformation_blur)
add_subdirectory(denoiser)
add_subdirectory(temporal_denoiser)
add_subdirectory(payload_annotation)

add_subdirectory(pick)
add_subdirectory(scene_edit)
add_subdirectory(uber)
