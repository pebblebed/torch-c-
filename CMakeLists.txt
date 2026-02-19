cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
list(APPEND CMAKE_PREFIX_PATH /home/ubuntu/.local/lib/python3.11/site-packages/torch/share/cmake)
project(charformer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(trainium "trainium.cpp" "dataset_dir.cpp")
target_link_libraries(trainium "${TORCH_LIBRARIES}")

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET trainium
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:trainium>)
endif (MSVC)


# Bring in gtest
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
target_link_libraries(tensor_tests gtest_main ${TORCH_LIBRARIES})
target_link_libraries(charformer_tests gtest_main ${TORCH_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(tensor_tests)
gtest_discover_tests(charformer_tests)

# Demos (built separately from tests)
add_subdirectory(demos)
