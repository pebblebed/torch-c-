add_executable(hello_trails "hello-trails.cpp")
target_link_libraries(hello_trails trails)

add_executable(mnist_compare "mnist-compare.cpp")
target_link_libraries(mnist_compare trails)

add_executable(charformer_shakespeare "charformer-shakespeare.cpp" "${CMAKE_SOURCE_DIR}/dataset_dir.cpp")
target_include_directories(charformer_shakespeare PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(charformer_shakespeare trails)

