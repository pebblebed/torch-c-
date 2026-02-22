add_executable(hello_trails "hello-trails.cpp")
target_include_directories(hello_trails PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(hello_trails "${TORCH_LIBRARIES}")

add_executable(mnist_compare "mnist-compare.cpp")
target_include_directories(mnist_compare PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(mnist_compare "${TORCH_LIBRARIES}")

add_executable(charformer_shakespeare "charformer-shakespeare.cpp" "${CMAKE_SOURCE_DIR}/dataset_dir.cpp")
target_include_directories(charformer_shakespeare PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(charformer_shakespeare "${TORCH_LIBRARIES}")

