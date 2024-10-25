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
    static std::string str() { return "()"; }
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
    static std::string str() { return std::to_string(N) + "," + next_t::str(); }
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

}

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
        return Tensor(torch::arange(start, numel(), 1).view({Dims...}));
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
    Tensor<> mean() {
        return { t_.mean() };
    }

    Tensor square() { return *this * *this; }
    Tensor operator*(Tensor other) { return { t_ * other.t() }; }
    Tensor operator*(float other) { return { t_ * other }; }
    Tensor operator/(float other) { return { t_ / other }; }
    Tensor operator+(Tensor other) { return { t_ + other.t() }; }
    Tensor operator-(Tensor other) { return { t_ - other.t() }; } 
    Tensor operator/(Tensor other) { return { t_ / other.t() }; }
    Tensor operator*(torch::Tensor other) { return { t_ * other }; }
    Tensor operator+(torch::Tensor other) { return { t_ + other }; }
    Tensor operator-(torch::Tensor other) { return { t_ - other }; } 
    Tensor operator/(torch::Tensor other) { return { t_ / other }; }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.t();
        return os;
    }

    std::string str() const {
        std::stringstream ss;
        ss << t();
        return ss.str();
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

namespace functional {
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

}
