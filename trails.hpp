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

namespace detail {

template<typename I=size_t, I ...N> struct int_sequence {};
// Specialization for 0-D
template<typename I>
struct int_sequence<I> {
    constexpr static I length = 0;
    constexpr static std::tuple<> values = std::tuple<>();
};

#if 0
template<int N, int... Dims>
struct int_sequence<N, Dims...> {
    constexpr static size_t length = 1 + int_sequence<Dims...>::length;
    constexpr static std::tuple<N, Dims...> values = std::tuple_cat(std::tuple<int>({N}), int_sequence<Dims...>::values);
    constexpr static int first = N;
    constexpr static int last = std::get<length - 1>(values);
    template<size_t i>
    struct get {
        constexpr static int value = std::tuple_element<i - 1, std::tuple<N, Dims...>>::value;
    };
};
#endif

}

#if 0
// Now a Tensor!
template<int... Dims> struct Tensor;

// Specialization for 0-D
template<>
struct Tensor<> {
    constexpr static int dims = 0;
    typedef dims_t = std::tuple<>;
};


// Specialization for concrete dimensions. -1 means "any size"
template<int N, int... Dims>
struct Tensor<N, Dims...> {
    typedef dims_t = std::tuple<int, 
    static auto size() {
        return std::tuple_cat(std::tuple<int>(N), NextDim::size());
    }

    using TupleType = std::tuple<N, Dims...>;
    constexpr static int k_FirstDim = N;
    constexpr static int k_LastDim = std::tuple_element<std::tuple_size<TupleType>::value - 1, TupleType>::value;
    using NextOuterTypeDim = Tensor<N, 
    using NextDim = Tensor<Dims...>;
    constexpr static int dims = 1 + NextDim::dims;

    Tensor()
    : t_(torch::randn({N, Dims...})) {}
    Tensor(torch::Tensor t)
    : t_(t) {
        assert(compare_sizes(t.sizes()));
    }

    NextDim& operator[](int i) {
        return NextDim(t_[i]);
    }
    private:
    torch::Tensor t_;

    bool compare_sizes(torch::IntArrayRef sizes) {
        if (sizes.size() != dims) {
            return false;
        }
        if (N != -1 && sizes[0] != N) {
            return false;
        }
        return NextDim::compare_sizes(sizes.slice(1));
    }
};
#endif
}
