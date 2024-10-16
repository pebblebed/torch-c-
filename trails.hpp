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

template<typename I=size_t, I ...N> struct val_sequence {};
// Specialization for 0-D
template<typename V>
struct val_sequence<V> {
    constexpr static size_t length = 0;
    typedef std::tuple<> tuple_t;
    constexpr static tuple_t values = tuple_t();
};

template<typename V, V N, V... Rest>
struct val_sequence<V, N, Rest...> {
    typedef val_sequence<V, Rest...> next_t;
    constexpr static size_t length = sizeof...(Rest) + 1;
    // Utility to cons up a tuple type
    template<typename ... input_t>
    using tuple_cat_t = decltype(std::tuple_cat(std::declval<input_t>()...));
    typedef tuple_cat_t<std::tuple<V>, typename next_t::tuple_t> tuple_t;
    constexpr static tuple_t values = std::tuple_cat(std::tuple<V>(N), val_sequence<V, Rest...>::values);

    template<int64_t i>
    constexpr static V get() {
        return std::get<size_t(i)>(values);
    }
};

// Primary template
template<typename Seq, int64_t i>
struct compare_sizes_helper_t {
    static bool compare(const Seq& seq, torch::IntArrayRef sizes) {
        return seq.template get<i>() == sizes[i] && compare_sizes_helper_t<Seq, i - 1>::compare(seq, sizes);
    }
};

// Specialization for i = -1
template<typename Seq>
struct compare_sizes_helper_t<Seq, -1> {
    static bool compare(const Seq&, torch::IntArrayRef) { return true; }
};

// Specialization for i = 0
template<typename Seq>
struct compare_sizes_helper_t<Seq, 0> {
    static bool compare(const Seq& seq, torch::IntArrayRef sizes) {
        return seq.template get<0>() == sizes[0];
    }
};

}

// Specialization for concrete dimensions. -1 means "any size"
template<int ...Dims>
struct Tensor {
    using seq_t = detail::val_sequence<size_t, Dims...>;
    constexpr static auto dims = seq_t::length;
    constexpr static auto size = seq_t::values;

    Tensor()
    : t_(torch::randn({Dims...})) {}
    Tensor(torch::Tensor t)
    : t_(t) {
        assert(compare_sizes(t.sizes()));
    }

    static Tensor randn() {
        return Tensor(torch::randn({Dims...}));
    }

    // Public for testing
    static bool compare_sizes(torch::IntArrayRef sizes) {
        if (sizes.size() != dims) {
            return false;
        }
        return detail::compare_sizes_helper_t<seq_t, int64_t(dims) - 1>::compare(seq_t(), sizes);
    }
 
    private:
    torch::Tensor t_;
};

}
