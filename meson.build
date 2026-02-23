# SPDX-License-Identifier: BSD-3-Clause

project(
  'trails',
  'cpp',
  version         : '0.1.0',
  license         : 'BSD-3-Clause',
  meson_version   : '>=1.1',
  default_options : ['cpp_std=c++20'],
)

# ---------------------------------------------------------------------------
# Find LibTorch via its CMake config files.
#
# LibTorch ships CMake package-config files but no native Meson wrap or
# pkg-config.  Meson's cmake dependency method bridges the gap.
#
# If torch is installed via pip (PyTorch), you may need:
#   CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
#       meson setup builddir
# ---------------------------------------------------------------------------
torch_dep = dependency('Torch', method : 'cmake')

# ---------------------------------------------------------------------------
# trails — header-only declared dependency
# ---------------------------------------------------------------------------
trails_inc = include_directories('include')

trails_dep = declare_dependency(
  include_directories : trails_inc,
  dependencies        : [torch_dep],
)

# ---------------------------------------------------------------------------
# Install headers (preserving include/trails/ subdirectory)
# ---------------------------------------------------------------------------
install_headers(
  'include/trails/trails.hpp',
  'include/trails/trails_nn.hpp',
  subdir : 'trails',
)

# ---------------------------------------------------------------------------
# Tests (opt-in)
# ---------------------------------------------------------------------------
if get_option('tests')
  gtest_dep = dependency('gtest_main', required : true)

  tensor_tests = executable(
    'tensor_tests',
    'tests/tensor_tests.cpp',
    dependencies : [trails_dep, gtest_dep],
  )

  charformer_tests = executable(
    'charformer_tests',
    'tests/charformer_tests.cpp',
    dependencies : [trails_dep, gtest_dep],
  )

  test('tensor_tests', tensor_tests)
  test('charformer_tests', charformer_tests)
endif

# ---------------------------------------------------------------------------
# Demos (opt-in)
# ---------------------------------------------------------------------------
if get_option('demos')
  executable(
    'hello_trails',
    'demos/hello-trails.cpp',
    dependencies : [trails_dep],
  )

  executable(
    'mnist_compare',
    'demos/mnist-compare.cpp',
    dependencies : [trails_dep],
  )

  executable(
    'charformer_shakespeare',
    'demos/charformer-shakespeare.cpp',
    'dataset_dir.cpp',
    include_directories : include_directories('.'),
    dependencies        : [trails_dep],
  )
endif

