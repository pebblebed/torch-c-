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

- [ ] `CRITICAL` Guard dataset sampling against modulo-by-zero when `file.size == n_ctx`.
  Current sampling does `% (file.size - n_ctx)`, which is zero in this edge case.
  Files:
  - `dataset_dir.hpp:43`
  - `dataset_dir.cpp:34`

- [ ] `HIGH` Fix `MMappedFile` constructor initializer-order bug.
  `lseek`/`mmap` currently use `fd` before checking whether `open()` succeeded.
  Files:
  - `dataset_dir.hpp:24`
  - `dataset_dir.hpp:25`
  - `dataset_dir.hpp:26`
  - `dataset_dir.hpp:28`

- [ ] `HIGH` Fix dataset directory traversal accounting.
  Recursive `traverse()` resets `total_size`, corrupting aggregate dataset size.
  Files:
  - `dataset_dir.cpp:56`

- [ ] `HIGH` Register submodules inside `ResNorm`.
  `layer` and `norm` are not registered as submodules, which can hide parameters from optimizers/checkpointing.
  Files:
  - `charformer.hpp:47`
  - `charformer.hpp:49`
  - `charformer.hpp:50`

- [ ] `HIGH` Harden device handling for inference/training paths.
  `forward(std::string)` creates CPU tensors regardless of model device; demo hard-forces `.mps()` without fallback.
  Files:
  - `charformer.hpp:215`
  - `charformer.hpp:225`
  - `demos/charformer-shakespeare.cpp:291`

- [ ] `MEDIUM` Replace release-stripped `assert` checks with runtime validation for critical paths.
  Files:
  - `charformer.hpp:143`
  - `charformer.hpp:150`

- [ ] `MEDIUM` Reduce invariant bypass surface from raw tensor exposure.
  Typed shape guarantees are easy to circumvent via public raw tensor access and untyped overloads.
  Files:
  - `include/trails/trails.hpp:269`
  - `include/trails/trails.hpp:339`
  - `include/trails/trails_nn.hpp:163`

- [ ] `MEDIUM` Improve production-readiness test strategy.
  Existing tests are broad, but accelerator tests are skipped without hardware and there is no soak/fault-injection coverage.
  Files:
  - `tests/charformer_tests.cpp:292`
  - `tests/charformer_tests.cpp:336`

- [ ] `LOW` Fix format-specifier mismatches in debug utility logging.
  Build emits warnings because `%zu` is used with `int64_t` values from LibTorch tensor dimensions/sizes.
  Files:
  - `util.hpp:11`
  - `util.hpp:14`
