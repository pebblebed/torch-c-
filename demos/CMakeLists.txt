add_executable(hello_trails "hello-trails.cpp")
target_include_directories(hello_trails PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(hello_trails "${TORCH_LIBRARIES}")

