# WIP: Mission-Critical Hardening Backlog

Date: 2026-02-26

## Open Items

- [x] `CRITICAL` Fix CharFormer training objective mismatch.
  Model outputs `log_softmax`, then training calls `cross_entropy` (which applies `log_softmax` again).
  Resolution: introduced shared `language_model_loss()` and switched it to `nll_loss` over flattened log-probs/targets; wired all training paths to use it.
  Files:
  - `charformer.hpp:210`
  - `trainium.cpp:68`
  - `demos/charformer-shakespeare.cpp:135`

- [x] `CRITICAL` Guard dataset sampling against modulo-by-zero when `file.size == n_ctx`.
  Current sampling does `% (file.size - n_ctx)`, which is zero in this edge case.
  Resolution: switched sampling window math to `file.size - n_ctx + 1`, added zero-window guard in `get()`, and added regression coverage for exact-context files.
  Note: this also fixed an off-by-one undercount in available samples.
  Files:
  - `dataset_dir.cpp:29`
  - `dataset_dir.cpp:40`
  - `dataset_dir.cpp:50`
  - `tests/dataset_tests.cpp:28`

- [x] `HIGH` Fix `MMappedFile` constructor initializer-order bug.
  `lseek`/`mmap` currently use `fd` before checking whether `open()` succeeded.
  Resolution: reordered constructor flow to validate `open()` first, preserved original errno in error messages, and added regression coverage for missing-path diagnostics.
  Files:
  - `dataset_dir.hpp:23`
  - `dataset_dir.hpp:36`
  - `dataset_dir.hpp:50`
  - `tests/dataset_tests.cpp:49`

- [x] `HIGH` Fix dataset directory traversal accounting.
  Recursive `traverse()` resets `total_size`, corrupting aggregate dataset size.
  Resolution: initialize `total_size` once in constructor and remove recursive resets; added regression coverage for sibling subdirectory accumulation.
  Files:
  - `dataset_dir.cpp:49`
  - `dataset_dir.cpp:55`
  - `tests/dataset_tests.cpp:79`

- [x] `HIGH` Register submodules inside `ResNorm`.
  `layer` and `norm` are not registered as submodules, which can hide parameters from optimizers/checkpointing.
  Resolution: converted `ResNorm` children to registered `shared_ptr` modules and added regression coverage asserting parameter visibility.
  Files:
  - `charformer.hpp:48`
  - `charformer.hpp:54`
  - `tests/charformer_tests.cpp:82`

- [x] `HIGH` Harden device handling for inference/training paths.
  `forward(std::string)` now builds tensors on the model's active device, model `.mps()`/`.cuda()` calls no longer hard-fail when backends are unavailable, and demo/device selection uses best-available fallback.
  Resolution: added `best_available_device()` / `module_device()` / `move_to_best_available_device()`, guarded accelerator moves across model blocks, and fixed string-byte conversion to avoid negative indices for extended bytes.
  Regression tests:
  - `CharformerTests.CharFormerMpsFallbackWhenUnavailable`
  - `CharformerTests.CharFormerStringForwardExtendedBytes`
  Files:
  - `charformer.hpp:33`
  - `charformer.hpp:65`
  - `charformer.hpp:262`
  - `demos/charformer-shakespeare.cpp:288`
  - `tests/charformer_tests.cpp:89`

- [x] `MEDIUM` Replace release-stripped `assert` checks with runtime validation for critical paths.
  Resolution: replaced assert-only guards in positional encoding helpers with explicit runtime validation and deterministic exceptions (input rank/shape constraints and argument sanity checks).
  Regression tests:
  - `CharformerTests.ApplyPosEncodingRejectsNon4DInput`
  Files:
  - `charformer.hpp:184`
  - `charformer.hpp:208`
  - `tests/charformer_tests.cpp:29`

- [x] `MEDIUM` Reduce invariant bypass surface from raw tensor exposure.
  Resolution: hardened `Tensor op torch::Tensor` overloads to reject broadcast/shape-mismatched raw operands, removed duplicate `BatchTensor::data()` raw escape hatch, and added deterministic `Embedding::forward` index dtype validation.
  Regression tests:
  - `TensorTest.EmbeddingRejectsFloatIndicesWithRuntimeError`
  - `ErrorHandlingTest.TensorRawTensorOperatorRejectsBroadcastShapes`
  Files:
  - `include/trails/trails.hpp:339`
  - `include/trails/trails.hpp:433`
  - `include/trails/trails_nn.hpp:163`
  - `tests/tensor_tests.cpp:716`
  - `tests/tensor_tests.cpp:2121`

- [x] `MEDIUM` Improve production-readiness test strategy.
  Resolution: added non-skipping accelerator contract tests that run on any host and assert Torch-consistent behavior (throw when backend unavailable, move parameters when available).
  Regression tests:
  - `BatchAgnosticTest.LinearCudaMatchesTorchAvailabilityContract`
  - `BatchAgnosticTest.LinearMpsMatchesTorchAvailabilityContract`
  Files:
  - `tests/tensor_tests.cpp:2371`
  - `tests/tensor_tests.cpp:2383`

- [ ] `MEDIUM` Add soak/fault-injection coverage for long-running reliability.
  Current suite is broad but still lacks long-horizon stress and injected-failure scenarios (OOM/device reset/restart/retry paths).
  Files:
  - `tests/charformer_tests.cpp:352`
  - `tests/tensor_tests.cpp:2357`

- [ ] `LOW` Fix format-specifier mismatches in debug utility logging.
  Build emits warnings because `%zu` is used with `int64_t` values from LibTorch tensor dimensions/sizes.
  Files:
  - `util.hpp:11`
  - `util.hpp:14`
