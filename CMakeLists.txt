cmake_minimum_required(VERSION 3.18)
project(
    OptiX_Utility_cmake
    VERSION 1.0
    DESCRIPTION "OptiX 7 Lightweight Wrapper Library"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
find_package(CUDAToolkit 11.6 REQUIRED)
find_package(OptiX74)
include("copy_files")
include("nvcuda_compile_ptx")

if (OptiX74_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 7.4 not found.")
endif()



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

# END:JP: プロジェクト間で共通のソースコードのリスト作成。
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# https://stackoverflow.com/questions/45092198/cmake-how-do-i-change-properties-on-subdirectory-project-targets

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットを再帰的に取得する。
function(get_all_targets _result _dir)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        get_all_targets(${_result} "${_subdir}")
    endforeach()
    get_property(_sub_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットにフォルダーを設定する。
function(set_project_group _folder_name _dir)
    get_all_targets(_targets "${_dir}")
    foreach(_target IN LISTS _targets)
        # message(${_target})
        set_target_properties(
            ${_target}
            PROPERTIES FOLDER "${_folder_name}"
        )
    endforeach()
endfunction()

# END: JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# ----------------------------------------------------------------

# Assimp
option(ASSIMP_BUILD_ASSIMP_TOOLS "If the supplementary tools for Assimp are built in addition to the library." OFF)
option(ASSIMP_BUILD_SAMPLES "If the official samples are built as well (needs Glut)." OFF)
option(ASSIMP_BUILD_TESTS "If the test suite for Assimp is built in addition to the library." OFF)
option(ASSIMP_INSTALL "Disable this if you want to use assimp as a submodule." OFF)
if(MSVC)
    option(ASSIMP_BUILD_ASSIMP_VIEW "If the Assimp view tool is built. (requires DirectX)" OFF)
endif()
add_subdirectory(ext/assimp)
set_project_group("External" "ext/assimp")

# GoogleTest
option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
add_subdirectory(ext/googletest)
set_project_group("External" "ext/googletest")

option(OPTIXU_BUILD_SAMPLES "Build sample projects for OptiX Utility." ON)
option(OPTIXU_BUILD_TESTS "Build test projects for OptiX Utility." ON)

if(OPTIXU_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
if(OPTIXU_BUILD_TESTS)
    add_subdirectory(tests/optixu_test)
endif()