Trails: Torch-on-Rails. This is a quality-of-life improvement for
the working machine-learning engineer: no more runtime tensor
mismatches. Tensor dimensionality is carried around in type
information, and preserved in all changes.

## C++ Philosophy

This is meant to be a single-header library that safely mixes in with many other C++
programs. We maintain hygiene around namespaces, all of which need to be under the project's
main namespace of `trails`.

We are careful about layering. `trails` itself is a numeric library, comparable to numpy or
torch. `trails::nn` is the neural network library layered on top of it. We never assume the
presence of `trails::nn` in `trails`-qua-`trails`.

We are careful about warnings. We don't check them in if we can help it.

## Pre-checkin ritual

1. *Does it build without warnings or errors*?
2. *Do all tests pass?*
3. *What fixing a bug, do we have a test for the new, correct behavior?*
4. *When introducing a feature, have we tried to test the coarse happy and unhappy paths through the feature*?

When all of the above are in a desirable state, we can build, run the tests, and commit.
