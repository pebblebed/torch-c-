cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
list(APPEND CMAKE_PREFIX_PATH /home/ubuntu/miniconda3/envs/CraftsMan/lib/python3.8/site-packages/torch/share/cmake)
project(diffusion3d-c++)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(trainium "trainium.cpp" "dataset_dir.cpp")
target_link_libraries(trainium "${TORCH_LIBRARIES}")
set_property(TARGET trainium PROPERTY CXX_STANDARD 17)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET trainium
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:trainium>)
endif (MSVC)
