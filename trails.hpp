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

// Zero-length specialization
template<typename V>
struct val_sequence<V> {
    constexpr static size_t length = 0;
    typedef std::tuple<> tuple_t;
    constexpr static tuple_t values = tuple_t();
    static std::string str() { return "()"; }

    class val_sequence_iterator { };
    constexpr static val_sequence_iterator begin() { return end(); }
    constexpr static val_sequence_iterator end() { return val_sequence_iterator{}; }

    // get/set not defined for 0-D
    template<typename S2>
    struct equals {
        static constexpr bool value = S2::length == 0;
    };

    template<size_t I>
    struct get {
        static_assert(I < length, "get index out of bounds");
        // Not reached
        static constexpr V value = V {};
    };
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

    template<size_t I>
    struct get : std::conditional_t<I == 0,
        std::integral_constant<V, N>,
        typename next_t::template get<I - 1>> { };


    // Helper to prepend a value to a val_sequence type
    template<V head, typename Tail>
    struct prepend;

    template<V head, V... tail_values>
    struct prepend<head, val_sequence<V, tail_values...>> {
        using type = val_sequence<V, head, tail_values...>;
    };

    template<int64_t i, V v>
    struct set_dim {
        using type = typename prepend<N, typename next_t::template set_dim<i - 1, v>::type>::type;
    };

    template<V v>
    struct set_dim<0, v> {
        using type = val_sequence<V, v, Rest...>;
    };

    template<typename S2>
    struct equals {
        static constexpr bool value = S2::length == length &&
            N == S2::template get<0>::value &&
            next_t::template equals<typename S2::next_t>::value;
    };

    static std::string str() { return std::to_string(N) + "," + next_t::str(); }
    struct val_sequence_iterator {
        V operator*() { return get<0>::value; }
        val_sequence_iterator& operator++() {
            return next_t::begin();
        }
    };
    constexpr static val_sequence_iterator begin() { return val_sequence_iterator(); }
    constexpr static val_sequence_iterator end() { return next_t::end(); }
};

// Primary template
template<typename Seq, int64_t i>
struct compare_sizes_helper_t {
    static bool compare(const Seq& seq, torch::IntArrayRef sizes) {
        return Seq::template get<i>::value == sizes[i] &&
        compare_sizes_helper_t<Seq, i - 1>::compare(seq, sizes);
    }
};

// Specialization for i = -1
template<typename Seq>
struct compare_sizes_helper_t<Seq, -1> {
    static bool compare(const Seq&, torch::IntArrayRef) { return true; }
};


std::string str(torch::IntArrayRef sizes) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < sizes.size(); ++i) {
        ss << sizes[i];
        if (i < sizes.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}


std::ostream& operator<<(std::ostream& os, torch::IntArrayRef sizes) {
    os << str(sizes);
    return os;
}

template<typename V, V ...Dims>
std::ostream& operator<<(std::ostream& os, const val_sequence<V, Dims...>& seq) {
    os << seq.str();
    return os;
}

template<typename TensorType, bool keepdim, int64_t ...reduceDims>
class ReduceDims { };

// ---- Shape-manipulation helpers for transpose, unsqueeze, squeeze, reshape, cat ----

// swap_dims<Seq, I, J>: swap dimensions at positions I and J via set_dim
template<typename Seq, size_t I, size_t J>
struct swap_dims {
    static constexpr auto vi = Seq::template get<I>::value;
    static constexpr auto vj = Seq::template get<J>::value;
    using step1 = typename Seq::template set_dim<I, vj>::type;
    using type = typename step1::template set_dim<J, vi>::type;
};

// insert_dim<Seq, Pos, Val>: insert a value at position Pos
template<typename Seq, size_t Pos, auto Val, typename Prefix = val_sequence<size_t>>
struct insert_dim;

// Pos == 0: insert Val before remaining elements
template<typename V, V Head, V... Tail, V Val, V... Prefix>
struct insert_dim<val_sequence<V, Head, Tail...>, 0, Val, val_sequence<V, Prefix...>> {
    using type = val_sequence<V, Prefix..., Val, Head, Tail...>;
};

// Pos > 0: move Head to Prefix and recurse
template<typename V, V Head, V... Tail, size_t Pos, V Val, V... Prefix>
struct insert_dim<val_sequence<V, Head, Tail...>, Pos, Val, val_sequence<V, Prefix...>> {
    using type = typename insert_dim<val_sequence<V, Tail...>, Pos - 1, Val, val_sequence<V, Prefix..., Head>>::type;
};

// Base case: empty sequence, Pos must be 0 — append Val
template<typename V, V Val, V... Prefix>
struct insert_dim<val_sequence<V>, 0, Val, val_sequence<V, Prefix...>> {
    using type = val_sequence<V, Prefix..., Val>;
};

// remove_dim<Seq, Pos>: remove dimension at position Pos
template<typename Seq, size_t Pos, typename Prefix = val_sequence<size_t>>
struct remove_dim;

// Pos == 0: skip Head, append Tail
template<typename V, V Head, V... Tail, V... Prefix>
struct remove_dim<val_sequence<V, Head, Tail...>, 0, val_sequence<V, Prefix...>> {
    using type = val_sequence<V, Prefix..., Tail...>;
};

// Pos > 0: move Head to Prefix and recurse
template<typename V, V Head, V... Tail, size_t Pos, V... Prefix>
struct remove_dim<val_sequence<V, Head, Tail...>, Pos, val_sequence<V, Prefix...>> {
    using type = typename remove_dim<val_sequence<V, Tail...>, Pos - 1, val_sequence<V, Prefix..., Head>>::type;
};

// replace_dim<Seq, Pos, NewVal>: replace dimension at Pos with NewVal
template<typename Seq, size_t Pos, auto NewVal>
struct replace_dim {
    using type = typename Seq::template set_dim<Pos, NewVal>::type;
};

} // namespace detail

// Forward declaration for seq_to_tensor
template<int ...Dims> struct Tensor;

namespace detail {

// seq_to_tensor: convert a val_sequence<size_t, ...> to a Tensor<int, ...> type
template<typename Seq>
struct seq_to_tensor;

template<size_t... Vals>
struct seq_to_tensor<val_sequence<size_t, Vals...>> {
    using type = Tensor<static_cast<int>(Vals)...>;
};

} // namespace detail

template<int ...Dims>
struct Tensor {
    using seq_t = detail::val_sequence<size_t, Dims...>;
    constexpr static size_t dim() { return seq_t::length; }
    constexpr static auto shape = seq_t::values;
    constexpr static size_t _numel(std::convertible_to<int> auto... dims) { return (... * static_cast<size_t>(dims)); }
    constexpr static size_t numel() { return _numel(Dims...); }
    template<size_t i>
    constexpr static size_t size = std::get<i>(shape);
    constexpr static decltype(shape) sizes() { return shape; }

    Tensor()
    : t_(torch::randn({Dims...})) {}
    Tensor(torch::Tensor t)
    : t_(t) {
        if (!compare_sizes(t.sizes())) {
            throw std::runtime_error("Tensor size mismatch: " + detail::str(t.sizes()) + " vs " + seq_t::str());
        }
    }

    torch::Tensor t() const { return t_; }
    template<typename T=float>
    const T* data_ptr() const { return t_.data_ptr<T>(); }

    static Tensor arange(float start=0) {
        return Tensor(torch::arange(start, static_cast<float>(numel()), 1).view({Dims...}));
    }

    static Tensor randn() {
        return Tensor(torch::randn({Dims...}));
    }

    static Tensor zeroes() {
        return Tensor(torch::zeros({Dims...}));
    }

    static Tensor ones() {
        return Tensor(torch::ones({Dims...}));
    }

    static bool compare_sizes(torch::IntArrayRef sizes) {
        if (seq_t::length != sizes.size()) {
            return false;
        }
        return detail::compare_sizes_helper_t<seq_t, int64_t(dim()) - 1>::compare(seq_t(), sizes);
    }
 
    Tensor cuda() { return { t_.cuda() }; }
    template<typename T=float>
    T item() const {
        return t_.item<T>();
    }
    Tensor abs() { return { t_.abs() }; }
    // XXX: all of these need some smart dim-removal technology.
    Tensor<> max() {
       return { t_.max() };
    }
    template<bool keepdim=false, int64_t ...reduceDims>
    detail::ReduceDims<Tensor, keepdim, reduceDims...>::tensor_t mean() {
        return { t_.mean(detail::ReduceDims<Tensor, keepdim, reduceDims...>::dims, keepdim) };
    }

    Tensor rsqrt() { return { t_.rsqrt() }; }
    Tensor square() { return *this * *this; }
    Tensor operator+(Tensor<Dims...> other) { return { t_ + other.t() }; }
    Tensor operator+(torch::Tensor other) { return { t_ + other }; }
    Tensor operator+(float other) { return { t_ + other }; }

    Tensor operator-(Tensor<Dims...> other) { return { t_ - other.t() }; } 
    Tensor operator-(torch::Tensor other) { return { t_ - other }; } 
    Tensor operator-(float other) { return { t_ - other }; }

    Tensor operator*(Tensor<Dims...> other) { return { t_ * other.t() }; }
    Tensor operator*(torch::Tensor other) { return { t_ * other }; }
    Tensor operator*(float other) { return { t_ * other }; }

    Tensor operator/(Tensor<Dims...> other) { return { t_ / other.t() }; }
    Tensor operator/(torch::Tensor other) { return { t_ / other }; }
    Tensor operator/(float other) { return { t_ / other }; }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.t();
        return os;
    }

    std::string str() const {
        std::stringstream ss;
        ss << t();
        return ss.str();
    }

    // transpose<dim0, dim1>(): swap two dimensions at compile time
    template<size_t dim0, size_t dim1>
    auto transpose() const {
        static_assert(dim0 < dim(), "transpose dim0 out of range");
        static_assert(dim1 < dim(), "transpose dim1 out of range");
        using new_seq = typename detail::swap_dims<seq_t, dim0, dim1>::type;
        using result_t = typename detail::seq_to_tensor<new_seq>::type;
        return result_t{ t_.transpose(dim0, dim1) };
    }

    // reshape<NewDims...>(): reshape with compile-time numel check
    template<int ...NewDims>
    Tensor<NewDims...> reshape() const {
        static_assert(numel() == Tensor<NewDims...>::numel(),
            "reshape: number of elements must match");
        return Tensor<NewDims...>{ t_.reshape({NewDims...}) };
    }

    // view<NewDims...>(): alias for reshape with contiguity
    template<int ...NewDims>
    Tensor<NewDims...> view() const {
        static_assert(numel() == Tensor<NewDims...>::numel(),
            "view: number of elements must match");
        return Tensor<NewDims...>{ t_.reshape({NewDims...}) };
    }

    // unsqueeze<D>(): insert a size-1 dimension at position D
    template<size_t D>
    auto unsqueeze() const {
        static_assert(D <= dim(), "unsqueeze dim out of range");
        using new_seq = typename detail::insert_dim<seq_t, D, size_t(1)>::type;
        using result_t = typename detail::seq_to_tensor<new_seq>::type;
        return result_t{ t_.unsqueeze(D) };
    }

    // squeeze<D>(): remove a size-1 dimension at position D
    template<size_t D>
    auto squeeze() const {
        static_assert(D < dim(), "squeeze dim out of range");
        static_assert(seq_t::template get<D>::value == 1,
            "squeeze: dimension must be size 1");
        using new_seq = typename detail::remove_dim<seq_t, D>::type;
        using result_t = typename detail::seq_to_tensor<new_seq>::type;
        return result_t{ t_.squeeze(D) };
    }

    private:
    torch::Tensor t_;
};

// Scalars on the outside
template<typename ftype = float, int ...Dims>
Tensor<Dims...>
operator*(ftype f, Tensor<Dims...> t) {
    return Tensor<Dims...>(f * t.t());
}

template<typename ftype = float, int ...Dims>
Tensor<Dims...>
operator+(ftype f, Tensor<Dims...> t) {
    return Tensor<Dims...>(f + t.t());
}

template<typename ftype = float, int ...Dims>
Tensor<Dims...>
operator-(ftype f, Tensor<Dims...> t) {
    return Tensor<Dims...>(f - t.t());
}

template<typename ftype = float, int ...Dims>
Tensor<Dims...>
operator/(ftype f, Tensor<Dims...> t) {
    return Tensor<Dims...>(f / t.t());
}

using Scalar = Tensor<>;

// matmul: 2D case Tensor<M,K> x Tensor<K,N> -> Tensor<M,N>
template<int M, int K, int N>
Tensor<M, N> matmul(Tensor<M, K> a, Tensor<K, N> b) {
    return Tensor<M, N>{ torch::matmul(a.t(), b.t()) };
}

// matmul: batched 3D case Tensor<B,M,K> x Tensor<B,K,N> -> Tensor<B,M,N>
template<int B, int M, int K, int N>
Tensor<B, M, N> matmul(Tensor<B, M, K> a, Tensor<B, K, N> b) {
    return Tensor<B, M, N>{ torch::matmul(a.t(), b.t()) };
}

namespace functional {

// cat<Dim>(a, b): concatenate two tensors along dimension Dim
// All dimensions must match except at Dim, where they are summed.
template<size_t Dim, int ...DimsA, int ...DimsB>
auto cat(Tensor<DimsA...> a, Tensor<DimsB...> b) {
    using SeqA = typename Tensor<DimsA...>::seq_t;
    using SeqB = typename Tensor<DimsB...>::seq_t;
    static_assert(SeqA::length == SeqB::length, "cat: tensors must have same number of dimensions");
    static_assert(Dim < SeqA::length, "cat: dim out of range");

    // Check all dims match except Dim (done via constexpr helper)
    constexpr auto check_dims = []<size_t... Is>(std::index_sequence<Is...>) {
        return ((Is == Dim || SeqA::template get<Is>::value == SeqB::template get<Is>::value) && ...);
    }(std::make_index_sequence<SeqA::length>{});
    static_assert(check_dims, "cat: all dimensions except cat dim must match");

    constexpr size_t new_dim_val = SeqA::template get<Dim>::value + SeqB::template get<Dim>::value;
    using new_seq = typename detail::replace_dim<SeqA, Dim, new_dim_val>::type;
    using result_t = typename detail::seq_to_tensor<new_seq>::type;
    return result_t{ torch::cat({a.t(), b.t()}, Dim) };
}

} // namespace functional (cat)

template<typename InputTensorType, typename OutputTensorType>
class Module : public torch::nn::Module {
    public:
    using input_t = InputTensorType;
    using output_t = OutputTensorType;
    virtual OutputTensorType forward(InputTensorType input) = 0;

    template<typename T>
    void register_parameter(std::string name, T value) {
        torch::nn::Module::register_parameter(name, value.t());
    }

    OutputTensorType operator()(InputTensorType input) {
        return forward(input);
    }
};

/*
 * Pipe operator for chaining modules together:
 *  auto y = x | module1 | module2 | ... | moduleN;
 */
template<typename InputTensorType, typename OutputTensorType>
OutputTensorType operator|(InputTensorType x, Module<InputTensorType, OutputTensorType>& m) {
    return m.forward(x);
}

template<typename InputTensorType, typename OutputTensorType,
      typename ...Layers>
class Sequential : public Module<InputTensorType, OutputTensorType> {
    std::tuple<Layers...> layers;
    public:
    Sequential(Layers&&... lyrs)
    : layers(std::forward<Layers>(lyrs)...)
    {}

    OutputTensorType forward(InputTensorType input) override {
        return std::apply([&input](auto &&...layers) {
            auto result = input;
            ((result = layers.forward(result)), ...);
            return result; }, layers);
    }
};

template<typename InputTensorType, typename OutputTensorType, typename TorchLayerType>
class TorchWrapperLayer : public Module<InputTensorType, OutputTensorType> {
    protected:
    TorchLayerType layer;
    public:
    TorchWrapperLayer(TorchLayerType& lyr)
    : layer(lyr) {}

    TorchWrapperLayer(auto ...args)
    : layer(args...) {}

    OutputTensorType forward(InputTensorType input) override {
        return { layer->forward(input.t()) };
    }
};

template<
    int B,
    int InDim,
    int OutDim>
class Linear : public TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear> {
    public:
    Linear()
    : TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear>(
        torch::nn::Linear(torch::nn::LinearOptions(InDim, OutDim)))
    {}

    Tensor<B, OutDim> forward(Tensor<B, InDim> input) override {
        return { TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear>::layer->forward(input.t()).reshape({B, OutDim}) };
    }
};

namespace functional {
/* conv1d.
 * See https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html#torch.nn.functional.conv1d
 */
template<
    int B,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int groups=1,
    int stride=1,
    int padding=0,
    int dilation=1>
Tensor<
    B,
    out_channels,
    ((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)>
conv1d(Tensor<B, in_channels, length> input,
       Tensor<out_channels, in_channels / groups, kernel_size> weights,
       std::optional<Tensor<B, out_channels, 1>> bias = std::nullopt) {
    return {
        torch::conv1d(input.t(), weights.t(),
                      bias ? bias->t() : torch::Tensor(),
                      /*stride*/ torch::IntArrayRef{stride},
                      /*padding*/ torch::IntArrayRef{padding},
                      /*dilation*/ torch::IntArrayRef{dilation},
                      /*groups*/ groups)
    };
}

template<
    int B,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int groups=1,
    int stride=1,
    int padding=0,
    int dilation=1>
Tensor<
    B,
    out_channels,
    ((input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1),
    ((input_width  + 2 * padding - dilation * (kernel_width  - 1) - 1) / stride + 1)>
conv2d(Tensor<B, in_channels, input_height, input_width> input,
       Tensor<out_channels, in_channels / groups, kernel_height, kernel_width> weights,
       std::optional<Tensor<out_channels>> bias = std::nullopt) {
    return { torch::conv2d(input.t(), weights.t(),
                           bias ? bias->t() : torch::Tensor(),
                           /*stride*/ torch::IntArrayRef{stride},
                           /*padding*/ torch::IntArrayRef{padding},
                           /*dilation*/ torch::IntArrayRef{dilation},
                           /*groups*/ groups)
    };
}

template<typename TensorType>
TensorType add(TensorType a, TensorType b) {
    return { a.t() + b.t() };
}

template<typename TensorType>
TensorType rms_norm(TensorType input, std::optional<TensorType> gamma = std::nullopt) {
    auto size = input.t().size();
    auto flat = input.t().view({input.t().size(0), -1});
    auto flat_y = input.t() * torch::rsqrt(input.t().square().mean(/*dim=*/0, /*keepdim=*/true) + 1e-6);
    auto gamma_mul = gamma ? flat_y * gamma : flat_y;
    return { flat_y.view(size) };
}

template<
    int B,
    int Length,
    int DictionarySize,
    int EmbeddingDim>
Tensor<B, EmbeddingDim>
project(Tensor<B, Length> input, Tensor<DictionarySize, EmbeddingDim> weights) {
    return torch::nn::functional::embedding(input.t(), weights.t());
}

}
namespace detail {

// Helper to convert tuple to array (must be before ReduceDims specializations)
template<typename T, typename Tuple, std::size_t... Is>
constexpr auto tuple_to_array_impl(const Tuple& t, std::index_sequence<Is...>) {
    return std::array<T, sizeof...(Is)>{(T)std::get<Is>(t)...};
}

template<typename T, typename Tuple>
constexpr auto tuple_to_array(const Tuple& t) {
    return tuple_to_array_impl<T>(t, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// No dims to reduce over and keepdim=false? -> return a Scalar
template<typename TensorType>
struct ReduceDims<TensorType, false> {
    using tensor_t = Scalar;
    static constexpr auto dims = torch::IntArrayRef{};
};

// Dims to reduce over but keepdim=false? Drop the selected dims
template<typename TensorType, int64_t ...reduceDims>
class ReduceDims<TensorType, false, reduceDims...> {
    template<size_t I, int64_t Dim, int64_t... Rest>
    struct DropDim {
        static constexpr bool should_keep = ((I != Dim) && ... && (I != Rest));
    };

    template<size_t... Is>
    static auto filter_dims(std::index_sequence<Is...>) {
        return std::integer_sequence<int64_t, 
            (DropDim<Is, reduceDims...>::should_keep ? TensorType::template size<Is> : 0)...
        >{};
    }

    template<int64_t... Filtered>
    static auto remove_zeros(std::integer_sequence<int64_t, Filtered...>) {
        return remove_zeros_impl<Filtered...>();
    }

private:
    // Helper to recursively filter out zeros from parameter pack
    template<int64_t First, int64_t... Rest>
    static auto remove_zeros_impl() {
        if constexpr (sizeof...(Rest) == 0) {
            // Base case: single element
            if constexpr (First != 0) {
                return std::integer_sequence<int64_t, First>{};
            } else {
                return std::integer_sequence<int64_t>{};
            }
        } else {
            // Recursive case: process first element and combine with rest
            auto rest_filtered = remove_zeros_impl<Rest...>();
            if constexpr (First != 0) {
                return prepend_to_sequence<First>(rest_filtered);
            } else {
                return rest_filtered;
            }
        }
    }

    // Helper to prepend a value to an integer sequence
    template<int64_t Value, int64_t... Seq>
    static auto prepend_to_sequence(std::integer_sequence<int64_t, Seq...>) {
        return std::integer_sequence<int64_t, Value, Seq...>{};
    }

    // Specialization for empty sequence
    static auto remove_zeros_impl() {
        return std::integer_sequence<int64_t>{};
    }

public:
    using filtered_dims = decltype(filter_dims(std::make_index_sequence<TensorType::dim()>{}));
    using final_dims = decltype(remove_zeros(filtered_dims{}));
    template<int64_t... FinalDims>
    static auto make_tensor(std::integer_sequence<int64_t, FinalDims...>) {
        return Tensor<FinalDims...>{};
    }

    using tensor_t = decltype(make_tensor(final_dims{}));
    constexpr static auto dims_array = tuple_to_array<int64_t>(tensor_t::sizes());
    constexpr static auto dims = torch::IntArrayRef(dims_array);
};

// Dims to reduce over but keepdim=true? 1-replace the selected dims
template<typename TensorType, int64_t ...reduceDims>
class ReduceDims<TensorType, true, reduceDims...> {
    template<size_t I, int64_t Dim, int64_t... Rest>
    struct OneOutDim {
        static constexpr bool should_keep = ((I != Dim) && ... && (I != Rest));
    };

    template<size_t... Is>
    static auto filter_dims(std::index_sequence<Is...>) {
        return std::integer_sequence<int64_t, 
            (OneOutDim<Is, reduceDims...>::should_keep ? TensorType::template size<Is> : 1)...
        >{};
    }
    using filtered_dims = decltype(filter_dims(std::make_index_sequence<TensorType::dim()>{}));

public:
    template<int64_t... FinalDims>
    static auto make_tensor(std::integer_sequence<int64_t, FinalDims...>) {
        return Tensor<FinalDims...>{};
    }

    using tensor_t = decltype(make_tensor(filtered_dims{}));
    constexpr static auto dims_array = tuple_to_array<int64_t>(tensor_t::sizes());
    constexpr static auto dims = torch::IntArrayRef(dims_array);
};

}

}
