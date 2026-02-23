cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Auto-detect PyTorch cmake path if not provided via -DCMAKE_PREFIX_PATH
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_CMAKE_RESULT
)
if(TORCH_CMAKE_RESULT EQUAL 0 AND TORCH_CMAKE_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
endif()

project(trails VERSION 0.1.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

# ---------------------------------------------------------------------------
# trails — header-only INTERFACE library
# ---------------------------------------------------------------------------
add_library(trails INTERFACE)
add_library(Trails::trails ALIAS trails)

target_include_directories(trails INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(trails INTERFACE cxx_std_20)
target_link_libraries(trails INTERFACE ${TORCH_LIBRARIES})

# ---------------------------------------------------------------------------
# Determine if this is a top-level build (standalone) or embedded via
# add_subdirectory / FetchContent.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TRAILS_IS_TOPLEVEL ON)
else()
    set(TRAILS_IS_TOPLEVEL OFF)
endif()

# ---------------------------------------------------------------------------
# Application target (only in standalone builds)
# ---------------------------------------------------------------------------
if(TRAILS_IS_TOPLEVEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    add_executable(trainium "trainium.cpp" "dataset_dir.cpp")
    target_link_libraries(trainium trails)

    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET trainium
                           POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           ${TORCH_DLLS}
                           $<TARGET_FILE_DIR:trainium>)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(TRAILS_BUILD_TESTS "Build trails tests" ${TRAILS_IS_TOPLEVEL})

if(TRAILS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(tensor_tests tests/tensor_tests.cpp)
    add_executable(charformer_tests tests/charformer_tests.cpp)
    target_link_libraries(tensor_tests gtest_main trails)
    target_link_libraries(charformer_tests gtest_main trails)

    include(GoogleTest)
    gtest_discover_tests(tensor_tests)
    gtest_discover_tests(charformer_tests)
endif()

# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
option(TRAILS_BUILD_DEMOS "Build trails demos" ${TRAILS_IS_TOPLEVEL})

if(TRAILS_BUILD_DEMOS)
    add_subdirectory(demos)
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(TARGETS trails EXPORT TrailsTargets)

install(DIRECTORY include/trails
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT TrailsTargets
        FILE TrailsTargets.cmake
        NAMESPACE Trails::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Trails)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TrailsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TrailsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Trails
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TrailsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TrailsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TrailsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Trails
)

# ---------------------------------------------------------------------------
# pkg-config
# ---------------------------------------------------------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/trails.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/trails.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/trails.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
