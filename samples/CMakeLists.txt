# ----------------------------------------------------------------
# JP: プロジェクト間で共通のソースコードのリスト作成。

file(
    GLOB COMMON_SOURCES
    "common/common.h"
    "common/common.cpp"
)
set(
    COMMON_SOURCES
    ${COMMON_SOURCES}
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

# TinyGLTF
file(
    GLOB TINY_GLTF_SOURCES
    "${CMAKE_SOURCE_DIR}/ext/tinygltf/tiny_gltf.h"
    "${CMAKE_SOURCE_DIR}/ext/tinygltf/tiny_gltf.cc"
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

file(
    GLOB COMMON_GUI_SOURCES
    "common/gui_common.h"
    "common/gui_common.cpp"
)
set(
    COMMON_GUI_SOURCES
    ${COMMON_GUI_SOURCES}
    ${COMMON_SHADERS}
    ${GL_SOURCES}
    ${IMGUI_SOURCES}
)

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
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/Zc:__cplusplus);
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/Zc:__cplusplus>"
    # if (compilerID == MSVC && compilerLanguage == CUDA) set(-Xcompiler "/wd 4819 /Zc:__cplusplus");
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4819 /Zc:__cplusplus>"
)
if(${CPP_VER} STREQUAL "c++17")
    target_compile_features(
        fakelib INTERFACE
        cxx_std_17
    )
elseif(${CPP_VER} STREQUAL "c++20")
    target_compile_features(
        fakelib INTERFACE
        cxx_std_20
    )
else()
    message(FATAL_ERROR "Invalid C++ standard.")
endif()



# データディレクトリのコピーを行うカスタムターゲットの定義。
set(SRC_DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(DST_DATA_DIR "${CMAKE_BINARY_DIR}/data")
file(GLOB_RECURSE SRC_DATA_FILES "${SRC_DATA_DIR}/*")
set(DATA_FILE_LIST "${CMAKE_BINARY_DIR}/data_file_list.txt")
add_custom_command(
    OUTPUT ${DATA_FILE_LIST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DST_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC_DATA_DIR} ${DST_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "${SRC_DATA_FILES}" > ${DATA_FILE_LIST}
    DEPENDS ${SRC_DATA_FILES}
    COMMENT "Copying directory and updating file list"
)
add_custom_target(__copy_data ALL DEPENDS ${DATA_FILE_LIST})
add_dependencies(fakelib __copy_data)



set(BASIC_SAMPLE_INDEX 0)

function(add_basic_sample SUBDIR)
    string(LENGTH "${BASIC_SAMPLE_INDEX}" STR_LENGTH)
    math(EXPR NUM_ZEROS "2 - ${STR_LENGTH}")
    if (NUM_ZEROS GREATER 0)
        string(REPEAT "0" ${NUM_ZEROS} ZEROS)
        set(BASIC_SAMPLE_INDEX_STR "${ZEROS}${BASIC_SAMPLE_INDEX}")
    else()
        set(BASIC_SAMPLE_INDEX_STR "${BASIC_SAMPLE_INDEX}")
    endif()

    add_subdirectory(${SUBDIR})

    math(EXPR BASIC_SAMPLE_INDEX "${BASIC_SAMPLE_INDEX} + 1")
    set(BASIC_SAMPLE_INDEX ${BASIC_SAMPLE_INDEX} PARENT_SCOPE)
endfunction()

add_basic_sample(single_gas)
add_basic_sample(single_level_instancing)
add_basic_sample(texture)
add_basic_sample(material_sets)
add_basic_sample(as_update)
add_basic_sample(custom_primitive)
add_basic_sample(curve_primitive)
add_basic_sample(sphere_primitive)
add_basic_sample(multi_level_instancing)
add_basic_sample(deformation_blur)
add_basic_sample(denoiser)
add_basic_sample(temporal_denoiser)
add_basic_sample(payload_annotation)
add_basic_sample(opacity_micro_map)
add_basic_sample(displacement_micro_map)

add_subdirectory(pick)
add_subdirectory(scene_edit)
add_subdirectory(uber)
