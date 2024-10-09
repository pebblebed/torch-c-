/* Copyright (c) 2024, Pebblebed Management, LLC. All rights reserved.
 * Author: Keith Adams <kma@pebblebed.com>
 *
 * Trails (tensors-on-rails) is a tensor library that checks shape compatibility
 * at compile time.
 */

#pragma once

#include <type_traits>
#include <tuple>
#include <utility>
#include <cstddef>

namespace trails {

class IndefiniteShapeDimension {};

template <int... N>
struct TensorDimension;

// Specialization for 0-D
template<>
struct TensorDimension<> {
    constexpr static int dims = 0;
    constexpr static std::tuple<> dims_tuple = std::tuple<>();
};

// Specialization for concrete dimensions
template<int N, int... Dims>
struct TensorDimension<N, Dims...> {
    constexpr static int value = N;
    using Next = TensorDimension<Dims...>;
    constexpr static int dims = 1 + Next::dims;
    constexpr static auto dims_tuple = std::tuple_cat(std::tuple<int>(N), Next::dims_tuple);
};



}