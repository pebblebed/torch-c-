/* Copyright (c) 2024, Pebblebed Management, LLC. All rights reserved.
 * Author: Keith Adams <kma@pebblebed.com>
 *
 * Trails (tensors-on-rails) is a tensor library that checks shape compatibility
 * at compile time.
 */

#pragma once

#include <torch/torch.h>
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


inline std::string str(torch::IntArrayRef sizes) {
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


inline std::ostream& operator<<(std::ostream& os, torch::IntArrayRef sizes) {
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

// Forward declarations
template<int ...Dims> struct Tensor;
template<int ...Dims> struct BatchTensor;

namespace detail {

// seq_to_tensor: convert a val_sequence<size_t, ...> to a Tensor<int, ...> type
template<typename Seq>
struct seq_to_tensor;

template<size_t... Vals>
struct seq_to_tensor<val_sequence<size_t, Vals...>> {
    using type = Tensor<static_cast<int>(Vals)...>;
};

// seq_to_batch_tensor: convert a val_sequence<size_t, ...> to a BatchTensor<int, ...> type
template<typename Seq>
struct seq_to_batch_tensor;

template<size_t... Vals>
struct seq_to_batch_tensor<val_sequence<size_t, Vals...>> {
    using type = BatchTensor<static_cast<int>(Vals)...>;
};

// tensor_to_batch: convert Tensor<Dims...> to BatchTensor<Dims...>
template<typename T> struct tensor_to_batch;
template<int ...Dims>
struct tensor_to_batch<Tensor<Dims...>> {
    using type = BatchTensor<Dims...>;
};

} // namespace detail

template<int ...Dims>
struct Tensor {
    using seq_t = detail::val_sequence<size_t, Dims...>;
    constexpr static size_t dim() { return seq_t::length; }
    constexpr static auto shape = seq_t::values;
    constexpr static size_t _numel(std::convertible_to<int> auto... dims) { return (... * static_cast<size_t>(dims)); }
    constexpr static size_t numel() {
        if constexpr (sizeof...(Dims) == 0) return 0;
        else return _numel(Dims...);
    }
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
    Tensor mps() { return { t_.to(torch::kMPS) }; }
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

    // Activation functions (shape-preserving)
    Tensor relu() { return { torch::relu(t_) }; }
    Tensor gelu() { return { torch::gelu(t_) }; }
    Tensor sigmoid() { return { torch::sigmoid(t_) }; }
    Tensor tanh() { return { torch::tanh(t_) }; }

    template<int64_t D>
    Tensor softmax() {
        static_assert(D >= 0 && D < dim(), "softmax: dim out of range");
        return { torch::softmax(t_, D) };
    }

    template<int64_t D>
    Tensor log_softmax() {
        static_assert(D >= 0 && D < dim(), "log_softmax: dim out of range");
        return { torch::log_softmax(t_, D) };
    }

    // Elementwise math (shape-preserving)
    Tensor exp() { return { t_.exp() }; }
    Tensor log() { return { t_.log() }; }
    Tensor sqrt() { return { t_.sqrt() }; }
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

    // unbatch: treat first dim as batch → BatchTensor<remaining dims>
    // Only valid when there is at least 1 dimension.
    auto unbatch() const {
        static_assert(dim() >= 1, "unbatch: need at least 1 dimension");
        // Remove the first dim from the sequence to get the mathematical dims
        using math_seq = typename detail::remove_dim<seq_t, 0>::type;
        using result_t = typename detail::seq_to_batch_tensor<math_seq>::type;
        return result_t{ t_ };
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

// ---- BatchTensor: a batch of identically-shaped tensors ----
// BatchTensor<Dims...> represents N independent Tensor<Dims...> objects.
// The mathematical shape is [Dims...]; the batch size is runtime.
// The underlying torch::Tensor has shape [batch_size, Dims...].

namespace detail {
// compare_sizes with offset: check t.sizes()[offset:] == {Dims...}
template<typename Seq, int64_t i, int64_t offset>
struct compare_sizes_offset_t {
    static bool compare(torch::IntArrayRef sizes) {
        return Seq::template get<i>::value == sizes[i + offset] &&
            compare_sizes_offset_t<Seq, i - 1, offset>::compare(sizes);
    }
};

template<typename Seq, int64_t offset>
struct compare_sizes_offset_t<Seq, -1, offset> {
    static bool compare(torch::IntArrayRef) { return true; }
};
} // namespace detail

template<int ...Dims>
struct BatchTensor {
    using inner_t = Tensor<Dims...>;
    using seq_t = typename inner_t::seq_t;
    constexpr static size_t math_dim() { return sizeof...(Dims); }

    // Construct from a dynamic torch::Tensor.
    // Validates that t.sizes()[1:] == {Dims...}; batch size is t.size(0).
    BatchTensor(torch::Tensor t) : t_(t) {
        constexpr int ndim = sizeof...(Dims);
        if (t.dim() != ndim + 1) {
            throw std::runtime_error(
                "BatchTensor dim mismatch: expected " + std::to_string(ndim + 1) +
                " but got " + std::to_string(t.dim()));
        }
        if (!detail::compare_sizes_offset_t<seq_t, int64_t(ndim) - 1, 1>::compare(t.sizes())) {
            throw std::runtime_error(
                "BatchTensor shape mismatch: got " + detail::str(t.sizes()) +
                " but expected [B, " + seq_t::str() + "]");
        }
    }

    int64_t batch_size() const { return t_.size(0); }
    torch::Tensor t() const { return t_; }
    torch::Tensor data() const { return t_; }

    // Promote to static tensor — runtime checks batch_size() == B
    template<int B>
    Tensor<B, Dims...> bind() const {
        if (batch_size() != B) {
            throw std::runtime_error(
                "BatchTensor::bind: expected batch_size " + std::to_string(B) +
                " but got " + std::to_string(batch_size()));
        }
        return Tensor<B, Dims...>(t_);
    }

    // Collapse batch via mean → static Tensor<Dims...>
    Tensor<Dims...> batch_mean() const {
        return Tensor<Dims...>(t_.mean(0));
    }

    // Reshape the mathematical dims (batch dim is unchanged)
    template<int ...NewDims>
    BatchTensor<NewDims...> reshape() const {
        static_assert((static_cast<int64_t>(Dims) * ...) == (static_cast<int64_t>(NewDims) * ...),
            "BatchTensor::reshape: number of elements must match");
        return BatchTensor<NewDims...>{ t_.reshape({batch_size(), NewDims...}) };
    }

    // Activations — element-wise, shape-preserving
    BatchTensor relu() const  { return { torch::relu(t_) }; }
    BatchTensor gelu() const  { return { torch::gelu(t_) }; }
    BatchTensor sigmoid() const { return { torch::sigmoid(t_) }; }
    BatchTensor tanh() const  { return { torch::tanh(t_) }; }

    template<int64_t D>
    BatchTensor softmax() const {
        static_assert(D >= 0 && D < (int64_t)math_dim(), "softmax: dim out of range");
        return { torch::softmax(t_, D + 1) };  // +1 for batch offset
    }

    template<int64_t D>
    BatchTensor log_softmax() const {
        static_assert(D >= 0 && D < (int64_t)math_dim(), "log_softmax: dim out of range");
        return { torch::log_softmax(t_, D + 1) };  // +1 for batch offset
    }

    // Elementwise binary ops — BatchTensor op BatchTensor
    BatchTensor operator+(const BatchTensor& o) const { return { t_ + o.t_ }; }
    BatchTensor operator-(const BatchTensor& o) const { return { t_ - o.t_ }; }
    BatchTensor operator*(const BatchTensor& o) const { return { t_ * o.t_ }; }
    BatchTensor operator/(const BatchTensor& o) const { return { t_ / o.t_ }; }

    // Scalar ops
    BatchTensor operator*(float s) const { return { t_ * s }; }
    BatchTensor operator/(float s) const { return { t_ / s }; }
    BatchTensor operator+(float s) const { return { t_ + s }; }
    BatchTensor operator-(float s) const { return { t_ - s }; }

    // Elementwise math (shape-preserving)
    BatchTensor exp() const { return { t_.exp() }; }
    BatchTensor log() const { return { t_.log() }; }
    BatchTensor sqrt() const { return { t_.sqrt() }; }

    // Transpose: swap mathematical dims D1 and D2 (NOT the batch dim)
    // The actual torch dims are D1+1 and D2+1 (offset by batch dim)
    template<int D1, int D2>
    auto transpose() const {
        static_assert(D1 >= 0 && D1 < (int)math_dim(), "transpose D1 out of range");
        static_assert(D2 >= 0 && D2 < (int)math_dim(), "transpose D2 out of range");
        using new_seq = typename detail::swap_dims<seq_t, D1, D2>::type;
        using result_t = typename detail::seq_to_batch_tensor<new_seq>::type;
        return result_t{ t_.transpose(D1 + 1, D2 + 1) };
    }

    // Shape-preserving unary ops
    BatchTensor rsqrt() const { return { t_.rsqrt() }; }
    BatchTensor square() const { return { t_ * t_ }; }
    BatchTensor abs() const { return { t_.abs() }; }
    BatchTensor cuda() const { return { t_.cuda() }; }
    BatchTensor mps() const { return { t_.to(torch::kMPS) }; }

    // Utility methods
    std::string str() const {
        std::stringstream ss;
        ss << t_;
        return ss.str();
    }

    template<typename T=float>
    const T* data_ptr() const { return t_.data_ptr<T>(); }

    // Reduction methods (reduce entire tensor to scalar)
    Tensor<> mean() const { return { t_.mean() }; }
    Tensor<> max() const { return { t_.max() }; }

    // Static factory methods with runtime batch size
    static BatchTensor randn(int batch_size) {
        return { torch::randn({batch_size, Dims...}) };
    }
    static BatchTensor zeroes(int batch_size) {
        return { torch::zeros({batch_size, Dims...}) };
    }
    static BatchTensor ones(int batch_size) {
        return { torch::ones({batch_size, Dims...}) };
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchTensor& bt) {
        os << "BatchTensor[B=" << bt.batch_size() << "] " << bt.t_;
        return os;
    }

private:
    torch::Tensor t_;
};

// Free function unbatch: Tensor<B, Dims...> → BatchTensor<Dims...>
template<int B, int ...Dims>
BatchTensor<Dims...> unbatch(Tensor<B, Dims...> t) {
    return BatchTensor<Dims...>(t.t());
}

// Scalar * BatchTensor
template<int ...Dims>
BatchTensor<Dims...> operator*(float s, BatchTensor<Dims...> bt) {
    return { bt.t() * s };
}

// Scalar + BatchTensor
template<int ...Dims>
BatchTensor<Dims...> operator+(float s, BatchTensor<Dims...> bt) {
    return { s + bt.t() };
}

// Scalar - BatchTensor
template<int ...Dims>
BatchTensor<Dims...> operator-(float s, BatchTensor<Dims...> bt) {
    return { s - bt.t() };
}

// Scalar / BatchTensor
template<int ...Dims>
BatchTensor<Dims...> operator/(float s, BatchTensor<Dims...> bt) {
    return { s / bt.t() };
}

// Broadcasting operators: BatchTensor op Tensor (broadcast static tensor over batch)
template<int ...Dims>
BatchTensor<Dims...> operator+(BatchTensor<Dims...> a, Tensor<Dims...> b) {
    return { a.t() + b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator-(BatchTensor<Dims...> a, Tensor<Dims...> b) {
    return { a.t() - b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator*(BatchTensor<Dims...> a, Tensor<Dims...> b) {
    return { a.t() * b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator/(BatchTensor<Dims...> a, Tensor<Dims...> b) {
    return { a.t() / b.t() };
}

// Reverse: Tensor op BatchTensor
template<int ...Dims>
BatchTensor<Dims...> operator+(Tensor<Dims...> a, BatchTensor<Dims...> b) {
    return { a.t() + b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator-(Tensor<Dims...> a, BatchTensor<Dims...> b) {
    return { a.t() - b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator*(Tensor<Dims...> a, BatchTensor<Dims...> b) {
    return { a.t() * b.t() };
}
template<int ...Dims>
BatchTensor<Dims...> operator/(Tensor<Dims...> a, BatchTensor<Dims...> b) {
    return { a.t() / b.t() };
}

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

// matmul: batched 4D case Tensor<B,H,M,K> x Tensor<B,H,K,N> -> Tensor<B,H,M,N>
template<int B, int H, int M, int K, int N>
Tensor<B, H, M, N> matmul(Tensor<B, H, M, K> a, Tensor<B, H, K, N> b) {
    return Tensor<B, H, M, N>{ torch::matmul(a.t(), b.t()) };
}

// matmul: Weight-sharing: BatchTensor<M,K> x Tensor<K,N> → BatchTensor<M,N>
template<int M, int K, int N>
BatchTensor<M, N> matmul(BatchTensor<M, K> a, Tensor<K, N> b) {
    return { torch::matmul(a.t(), b.t()) };
}

// matmul: Per-sample: BatchTensor<M,K> x BatchTensor<K,N> → BatchTensor<M,N>
template<int M, int K, int N>
BatchTensor<M, N> matmul(BatchTensor<M, K> a, BatchTensor<K, N> b) {
    return { torch::matmul(a.t(), b.t()) };
}

// scaled_dot_product_attention:
//   Q: Tensor<B, H, L, D>  (queries)
//   K: Tensor<B, H, S, D>  (keys)
//   V: Tensor<B, H, S, Dv> (values)
//   -> Tensor<B, H, L, Dv>
// Computes: softmax(Q @ K^T / sqrt(D)) @ V
template<int B, int H, int L, int S, int D, int Dv>
Tensor<B, H, L, Dv>
scaled_dot_product_attention(Tensor<B, H, L, D> Q,
                             Tensor<B, H, S, D> K,
                             Tensor<B, H, S, Dv> V) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    // Q @ K^T -> (B, H, L, S)
    auto scores = matmul(Q, K.template transpose<2, 3>()) * scale;
    // softmax over last dim (the S / key dimension)
    auto weights = scores.template softmax<3>();
    // weights @ V -> (B, H, L, Dv)
    return matmul(weights, V);
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
       std::optional<Tensor<out_channels>> bias = std::nullopt) {
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

// conv2d for BatchTensor: BatchTensor<InC, H, W> + weight Tensor<OutC, InC/G, KH, KW>
template<
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
BatchTensor<
    out_channels,
    ((input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1),
    ((input_width  + 2 * padding - dilation * (kernel_width  - 1) - 1) / stride + 1)>
conv2d(BatchTensor<in_channels, input_height, input_width> input,
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

// Activation functions (shape-preserving)
template<typename TensorType>
TensorType relu(TensorType input) { return input.relu(); }

template<typename TensorType>
TensorType gelu(TensorType input) { return input.gelu(); }

template<typename TensorType>
TensorType sigmoid(TensorType input) { return input.sigmoid(); }

template<typename TensorType>
TensorType tanh(TensorType input) { return input.tanh(); }

template<int64_t D, typename TensorType>
TensorType softmax(TensorType input) { return input.template softmax<D>(); }

template<int64_t D, typename TensorType>
TensorType log_softmax(TensorType input) { return input.template log_softmax<D>(); }

template<typename TensorType>
TensorType dropout(TensorType input, double p = 0.5, bool training = true) {
    return { torch::dropout(input.t(), p, training) };
}

// Elementwise math (shape-preserving)
template<typename TensorType>
TensorType exp(TensorType input) { return input.exp(); }

template<typename TensorType>
TensorType log(TensorType input) { return input.log(); }

template<typename TensorType>
TensorType sqrt(TensorType input) { return input.sqrt(); }

// ---- Pooling operations ----

/*
 * max_pool1d: 1D max pooling with compile-time shape propagation.
 * Input: Tensor<B, C, L> -> Output: Tensor<B, C, (L - KernelSize) / Stride + 1>
 */
template<
    int KernelSize,
    int Stride,
    int B,
    int C,
    int L>
Tensor<B, C, ((L - KernelSize) / Stride + 1)>
max_pool1d(Tensor<B, C, L> input) {
    return { torch::max_pool1d(input.t(),
                               /*kernel_size=*/{KernelSize},
                               /*stride=*/{Stride}) };
}

/*
 * max_pool2d: 2D max pooling with compile-time shape propagation.
 * Input: Tensor<B, C, H, W> -> Output: Tensor<B, C, (H-KH)/SH+1, (W-KW)/SW+1>
 */
template<
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int B,
    int C,
    int H,
    int W>
Tensor<B, C, ((H - KernelH) / StrideH + 1), ((W - KernelW) / StrideW + 1)>
max_pool2d(Tensor<B, C, H, W> input) {
    return { torch::max_pool2d(input.t(),
                               /*kernel_size=*/{KernelH, KernelW},
                               /*stride=*/{StrideH, StrideW}) };
}

// max_pool2d for BatchTensor: BatchTensor<C, H, W> → BatchTensor<C, outH, outW>
template<
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int C,
    int H,
    int W>
BatchTensor<C, ((H - KernelH) / StrideH + 1), ((W - KernelW) / StrideW + 1)>
max_pool2d(BatchTensor<C, H, W> input) {
    return { torch::max_pool2d(input.t(),
                               /*kernel_size=*/{KernelH, KernelW},
                               /*stride=*/{StrideH, StrideW}) };
}

/*
 * avg_pool1d: 1D average pooling with compile-time shape propagation.
 * Input: Tensor<B, C, L> -> Output: Tensor<B, C, (L - KernelSize) / Stride + 1>
 */
template<
    int KernelSize,
    int Stride,
    int B,
    int C,
    int L>
Tensor<B, C, ((L - KernelSize) / Stride + 1)>
avg_pool1d(Tensor<B, C, L> input) {
    return { torch::avg_pool1d(input.t(),
                               /*kernel_size=*/{KernelSize},
                               /*stride=*/{Stride}) };
}

/*
 * avg_pool2d: 2D average pooling with compile-time shape propagation.
 * Input: Tensor<B, C, H, W> -> Output: Tensor<B, C, (H-KH)/SH+1, (W-KW)/SW+1>
 */
template<
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int B,
    int C,
    int H,
    int W>
Tensor<B, C, ((H - KernelH) / StrideH + 1), ((W - KernelW) / StrideW + 1)>
avg_pool2d(Tensor<B, C, H, W> input) {
    return { torch::avg_pool2d(input.t(),
                               /*kernel_size=*/{KernelH, KernelW},
                               /*stride=*/{StrideH, StrideW}) };
}

// avg_pool2d for BatchTensor: BatchTensor<C, H, W> → BatchTensor<C, outH, outW>
template<
    int KernelH,
    int KernelW,
    int StrideH,
    int StrideW,
    int C,
    int H,
    int W>
BatchTensor<C, ((H - KernelH) / StrideH + 1), ((W - KernelW) / StrideW + 1)>
avg_pool2d(BatchTensor<C, H, W> input) {
    return { torch::avg_pool2d(input.t(),
                               /*kernel_size=*/{KernelH, KernelW},
                               /*stride=*/{StrideH, StrideW}) };
}

// ---- Flatten ----

} // namespace functional (for flatten detail helper)

namespace detail {

// Compile-time product of a range of elements in an array
template<int Start, int End, int ...Dims>
struct dim_product {
    static constexpr auto arr = std::array<int, sizeof...(Dims)>{Dims...};
    static constexpr int value = arr[Start] * dim_product<Start + 1, End, Dims...>::value;
};

template<int End, int ...Dims>
struct dim_product<End, End, Dims...> {
    static constexpr auto arr = std::array<int, sizeof...(Dims)>{Dims...};
    static constexpr int value = arr[End];
};

// Build the flattened shape: dims[0..Start) ++ product ++ dims[End+1..N)
template<int StartDim, int EndDim, int ...Dims>
struct flatten_result {
    static constexpr int ndim = sizeof...(Dims);
    static constexpr auto arr = std::array<int, ndim>{Dims...};
    static constexpr int flat_dim = dim_product<StartDim, EndDim, Dims...>::value;

    template<size_t... Pre, size_t... Post>
    static auto make(std::index_sequence<Pre...>, std::index_sequence<Post...>)
        -> Tensor<arr[Pre]..., flat_dim, arr[EndDim + 1 + Post]...>;

    using type = decltype(make(
        std::make_index_sequence<StartDim>{},
        std::make_index_sequence<ndim - EndDim - 1>{}));
};

} // namespace detail

namespace functional {

/*
 * flatten<StartDim, EndDim>: flatten dimensions from StartDim to EndDim (inclusive).
 * Compile-time shape computation.
 */
template<int StartDim, int EndDim, int ...Dims>
auto flatten(Tensor<Dims...> input) {
    constexpr int ndim = sizeof...(Dims);
    static_assert(StartDim >= 0 && StartDim < ndim, "flatten: StartDim out of range");
    static_assert(EndDim >= StartDim && EndDim < ndim, "flatten: EndDim out of range");
    using result_t = typename detail::flatten_result<StartDim, EndDim, Dims...>::type;
    return result_t{ torch::flatten(input.t(), StartDim, EndDim) };
}

/*
 * flatten<StartDim, EndDim> for BatchTensor: flatten mathematical dims.
 * StartDim/EndDim are over the mathematical (non-batch) dims.
 * The underlying torch dims are StartDim+1, EndDim+1 (offset by batch).
 */
template<int StartDim, int EndDim, int ...Dims>
auto flatten(BatchTensor<Dims...> input) {
    constexpr int ndim = sizeof...(Dims);
    static_assert(StartDim >= 0 && StartDim < ndim, "flatten: StartDim out of range");
    static_assert(EndDim >= StartDim && EndDim < ndim, "flatten: EndDim out of range");
    using inner_result_t = typename detail::flatten_result<StartDim, EndDim, Dims...>::type;
    using result_t = typename detail::tensor_to_batch<inner_result_t>::type;
    return result_t{ torch::flatten(input.t(), StartDim + 1, EndDim + 1) };
}

// ---- Functional linear ----

/*
 * linear: functional linear transformation.
 * 2D: Tensor<M, InDim> x Tensor<OutDim, InDim> -> Tensor<M, OutDim>
 */
template<int M, int InDim, int OutDim>
Tensor<M, OutDim>
linear(Tensor<M, InDim> input, Tensor<OutDim, InDim> weight,
       std::optional<Tensor<OutDim>> bias = std::nullopt) {
    return { torch::nn::functional::linear(input.t(), weight.t(),
             bias ? bias->t() : torch::Tensor()) };
}

/*
 * linear: functional linear transformation (batched 3D).
 * Tensor<B, SeqLen, InDim> x Tensor<OutDim, InDim> -> Tensor<B, SeqLen, OutDim>
 */
template<int B, int SeqLen, int InDim, int OutDim>
Tensor<B, SeqLen, OutDim>
linear(Tensor<B, SeqLen, InDim> input, Tensor<OutDim, InDim> weight,
       std::optional<Tensor<OutDim>> bias = std::nullopt) {
    return { torch::nn::functional::linear(input.t(), weight.t(),
             bias ? bias->t() : torch::Tensor()) };
}

/*
 * linear for BatchTensor: BatchTensor<InDim> x Tensor<OutDim, InDim> → BatchTensor<OutDim>
 */
template<int InDim, int OutDim>
BatchTensor<OutDim>
linear(BatchTensor<InDim> input, Tensor<OutDim, InDim> weight,
       std::optional<Tensor<OutDim>> bias = std::nullopt) {
    return { torch::nn::functional::linear(input.t(), weight.t(),
             bias ? bias->t() : torch::Tensor()) };
}

/*
 * linear for BatchTensor (3D): BatchTensor<SeqLen, InDim> x Tensor<OutDim, InDim> → BatchTensor<SeqLen, OutDim>
 */
template<int SeqLen, int InDim, int OutDim>
BatchTensor<SeqLen, OutDim>
linear(BatchTensor<SeqLen, InDim> input, Tensor<OutDim, InDim> weight,
       std::optional<Tensor<OutDim>> bias = std::nullopt) {
    return { torch::nn::functional::linear(input.t(), weight.t(),
             bias ? bias->t() : torch::Tensor()) };
}

// scaled_dot_product_attention (forwarding to trails:: free function)
template<int B, int H, int L, int S, int D, int Dv>
Tensor<B, H, L, Dv>
scaled_dot_product_attention(Tensor<B, H, L, D> Q,
                             Tensor<B, H, S, D> K,
                             Tensor<B, H, S, Dv> V) {
    return trails::scaled_dot_product_attention(Q, K, V);
}

// ---- Cross-entropy loss ----

/*
 * cross_entropy: static-batch version.
 * Tensor<B,C> logits + Tensor<B> integer labels → scalar torch::Tensor.
 */
template<int B, int C>
torch::Tensor cross_entropy(Tensor<B, C> input, Tensor<B> target) {
    return torch::nn::functional::cross_entropy(
        input.t(), target.t()
    );
}

/*
 * cross_entropy: dynamic-batch (BatchTensor) version.
 * BatchTensor<C> logits + torch::Tensor integer labels → scalar torch::Tensor.
 */
template<int C>
torch::Tensor cross_entropy(BatchTensor<C> input, torch::Tensor target) {
    return torch::nn::functional::cross_entropy(
        input.t(), target
    );
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

template<int64_t Query, int64_t... Dims>
struct contains_dim : std::bool_constant<((Query == Dims) || ...)> {};

template<int64_t Value, typename Seq>
struct prepend_sequence;

template<int64_t Value, int64_t... Seq>
struct prepend_sequence<Value, std::integer_sequence<int64_t, Seq...>> {
    using type = std::integer_sequence<int64_t, Value, Seq...>;
};

template<int64_t... Values>
struct filter_nonnegative_sequence;

template<>
struct filter_nonnegative_sequence<> {
    using type = std::integer_sequence<int64_t>;
};

template<int64_t First, int64_t... Rest>
struct filter_nonnegative_sequence<First, Rest...> {
    using tail_t = typename filter_nonnegative_sequence<Rest...>::type;
    using type = std::conditional_t<
        (First >= 0),
        typename prepend_sequence<First, tail_t>::type,
        tail_t>;
};

template<typename TensorType, int64_t... reduceDims>
consteval bool validate_reduce_dims() {
    constexpr auto ndim = static_cast<int64_t>(TensorType::dim());
    constexpr bool all_in_range = ((reduceDims >= 0 && reduceDims < ndim) && ...);
    if constexpr (!all_in_range) {
        return false;
    }

    constexpr std::array<int64_t, sizeof...(reduceDims)> dims = {reduceDims...};
    for (size_t i = 0; i < dims.size(); ++i) {
        for (size_t j = i + 1; j < dims.size(); ++j) {
            if (dims[i] == dims[j]) {
                return false;
            }
        }
    }
    return true;
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
    static_assert(validate_reduce_dims<TensorType, reduceDims...>(),
        "ReduceDims: reduction dimensions must be unique and within bounds");

    template<size_t... Is>
    static auto filter_dims(std::index_sequence<Is...>) {
        return std::integer_sequence<int64_t,
            (contains_dim<static_cast<int64_t>(Is), reduceDims...>::value
                ? -1
                : static_cast<int64_t>(TensorType::template size<Is>))...>{};
    }

public:
    using filtered_dims = decltype(filter_dims(std::make_index_sequence<TensorType::dim()>{}));
    template<int64_t... Filtered>
    static auto make_final_dims(std::integer_sequence<int64_t, Filtered...>) {
        using seq_t = typename filter_nonnegative_sequence<Filtered...>::type;
        return seq_t{};
    }
    using final_dims = decltype(make_final_dims(filtered_dims{}));

    template<int64_t... FinalDims>
    static auto make_tensor(std::integer_sequence<int64_t, FinalDims...>) {
        return Tensor<static_cast<int>(FinalDims)...>{};
    }

    using tensor_t = decltype(make_tensor(final_dims{}));
    constexpr static std::array<int64_t, sizeof...(reduceDims)> dims_array = {reduceDims...};
    constexpr static auto dims = torch::IntArrayRef(dims_array);
};

// Dims to reduce over but keepdim=true? 1-replace the selected dims
template<typename TensorType>
class ReduceDims<TensorType, true> {
    template<size_t... Is>
    static auto make_ones(std::index_sequence<Is...>) {
        return Tensor<((void)Is, 1)...>{};
    }

public:
    using tensor_t = decltype(make_ones(std::make_index_sequence<TensorType::dim()>{}));
    static constexpr auto dims = torch::IntArrayRef{};
};

template<typename TensorType, int64_t ...reduceDims>
class ReduceDims<TensorType, true, reduceDims...> {
    static_assert(validate_reduce_dims<TensorType, reduceDims...>(),
        "ReduceDims: reduction dimensions must be unique and within bounds");

    template<size_t... Is>
    static auto filter_dims(std::index_sequence<Is...>) {
        return std::integer_sequence<int64_t,
            (contains_dim<static_cast<int64_t>(Is), reduceDims...>::value
                ? 1
                : static_cast<int64_t>(TensorType::template size<Is>))...>{};
    }
    using filtered_dims = decltype(filter_dims(std::make_index_sequence<TensorType::dim()>{}));

public:
    template<int64_t... FinalDims>
    static auto make_tensor(std::integer_sequence<int64_t, FinalDims...>) {
        return Tensor<static_cast<int>(FinalDims)...>{};
    }

    using tensor_t = decltype(make_tensor(filtered_dims{}));
    constexpr static std::array<int64_t, sizeof...(reduceDims)> dims_array = {reduceDims...};
    constexpr static auto dims = torch::IntArrayRef(dims_array);
};

}

}
