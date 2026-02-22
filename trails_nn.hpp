/* Copyright (c) 2024, Pebblebed Management, LLC. All rights reserved.
 * Author: Keith Adams <kma@pebblebed.com>
 *
 * Trails nn module wrappers: typed wrappers around torch.nn modules
 * with compile-time shape checking.
 */

#pragma once

#include <torch/torch.h>
#include "trails.hpp"

namespace trails::nn {

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
    : layer(torch::nn::Module::register_module("layer", lyr)) {}

    TorchWrapperLayer(auto ...args)
    : layer(torch::nn::Module::register_module("layer", TorchLayerType(args...))) {}

    OutputTensorType forward(InputTensorType input) override {
        return { layer->forward(input.t()) };
    }
};

// Primary variadic template — specializations below for static-batch (3 params)
// and batch-agnostic (2 params).
template<int ...Dims>
class Linear;

template<
    int B,
    int InDim,
    int OutDim>
class Linear<B, InDim, OutDim> : public TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear> {
    public:
    Linear()
    : TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear>(
        torch::nn::Linear(torch::nn::LinearOptions(InDim, OutDim)))
    {}

    Tensor<B, OutDim> forward(Tensor<B, InDim> input) override {
        return { TorchWrapperLayer<Tensor<B, InDim>, Tensor<B, OutDim>, torch::nn::Linear>::layer->forward(input.t()).reshape({B, OutDim}) };
    }
};

/*
 * Batch-agnostic Linear: BatchTensor<InDim> → BatchTensor<OutDim>.
 * Works with any batch size at runtime.
 */
template<int InDim, int OutDim>
class Linear<InDim, OutDim> : public torch::nn::Module {
    torch::nn::Linear inner_;
public:
    Linear()
    : inner_(torch::nn::Linear(torch::nn::LinearOptions(InDim, OutDim)))
    {
        register_module("linear", inner_);
    }

    BatchTensor<OutDim> forward(BatchTensor<InDim> input) {
        return BatchTensor<OutDim>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * Batch-agnostic Conv2d: BatchTensor<InC, H, W> → BatchTensor<OutC, OutH, OutW>.
 * Works with any batch size at runtime. Forward is templated on spatial dims H, W
 * so it works with any input spatial size.
 *
 * Template params:
 *   InC      - input channels
 *   OutC     - output channels
 *   KH       - kernel height
 *   KW       - kernel width
 *   Stride   - stride (default 1)
 *   Padding  - padding (default 0)
 *   Dilation - dilation (default 1)
 *   Groups   - groups (default 1)
 */
template<int InC, int OutC, int KH, int KW, int Stride=1, int Padding=0, int Dilation=1, int Groups=1>
class Conv2d : public torch::nn::Module {
    torch::nn::Conv2d inner_;
public:
    Conv2d()
    : inner_(torch::nn::Conv2d(torch::nn::Conv2dOptions(InC, OutC, {KH, KW})
        .stride(Stride).padding(Padding).dilation(Dilation).groups(Groups)))
    {
        register_module("conv2d", inner_);
    }

    template<int H, int W>
    BatchTensor<
        OutC,
        ((H + 2 * Padding - Dilation * (KH - 1) - 1) / Stride + 1),
        ((W + 2 * Padding - Dilation * (KW - 1) - 1) / Stride + 1)>
    forward(BatchTensor<InC, H, W> input) {
        constexpr int OutH = (H + 2 * Padding - Dilation * (KH - 1) - 1) / Stride + 1;
        constexpr int OutW = (W + 2 * Padding - Dilation * (KW - 1) - 1) / Stride + 1;
        return BatchTensor<OutC, OutH, OutW>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * LayerNorm: primary variadic template — specializations below for
 * static-batch (2+ params: B, Dims...) and batch-agnostic (1 param: Dim).
 */
template<int ...AllDims>
class LayerNorm;

/*
 * Static-batch LayerNorm: wraps torch::nn::LayerNorm.
 * Normalizes over the last sizeof...(Dims)+1 dimensions.
 * Input/output shape: Tensor<B, FirstDim, RestDims...>
 * Requires at least 2 template params (B + at least one normalization dim).
 */
template<int B, int FirstDim, int ...RestDims>
class LayerNorm<B, FirstDim, RestDims...> : public TorchWrapperLayer<Tensor<B, FirstDim, RestDims...>, Tensor<B, FirstDim, RestDims...>, torch::nn::LayerNorm> {
    using InputType = Tensor<B, FirstDim, RestDims...>;
    using Base = TorchWrapperLayer<InputType, InputType, torch::nn::LayerNorm>;
public:
    LayerNorm()
    : Base(torch::nn::LayerNorm(torch::nn::LayerNormOptions({FirstDim, RestDims...})))
    {}
};

/*
 * Batch-agnostic LayerNorm: BatchTensor<Dim> → BatchTensor<Dim>.
 * Works with any batch size at runtime. Single normalization dimension.
 */
template<int Dim>
class LayerNorm<Dim> : public torch::nn::Module {
    torch::nn::LayerNorm inner_;
public:
    LayerNorm()
    : inner_(torch::nn::LayerNorm(torch::nn::LayerNormOptions({Dim})))
    {
        register_module("layer_norm", inner_);
    }

    BatchTensor<Dim> forward(BatchTensor<Dim> input) {
        return BatchTensor<Dim>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * BatchLayerNorm: batch-agnostic multi-dim LayerNorm.
 * BatchTensor<Dims...> → BatchTensor<Dims...>.
 * Normalizes over all mathematical dimensions (the trailing dims after batch).
 * Uses torch::nn::LayerNorm({Dims...}) internally.
 *
 * This is a separate class from LayerNorm to avoid arity ambiguity with
 * the static-batch LayerNorm<B, FirstDim, RestDims...>.
 */
template<int ...Dims>
class BatchLayerNorm : public torch::nn::Module {
    torch::nn::LayerNorm inner_;
public:
    BatchLayerNorm()
    : inner_(torch::nn::LayerNorm(torch::nn::LayerNormOptions({Dims...})))
    {
        register_module("layer_norm", inner_);
    }

    BatchTensor<Dims...> forward(BatchTensor<Dims...> input) {
        return BatchTensor<Dims...>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * BatchNorm1d: wraps torch::nn::BatchNorm1d.
 * Input/output shape: Tensor<B, C, L>
 */
template<int B, int C, int L>
class BatchNorm1d : public TorchWrapperLayer<Tensor<B, C, L>, Tensor<B, C, L>, torch::nn::BatchNorm1d> {
    using InputType = Tensor<B, C, L>;
    using Base = TorchWrapperLayer<InputType, InputType, torch::nn::BatchNorm1d>;
public:
    BatchNorm1d()
    : Base(torch::nn::BatchNorm1d(torch::nn::BatchNormOptions(C)))
    {}
};

/*
 * BatchNorm2d: wraps torch::nn::BatchNorm2d.
 * Input/output shape: Tensor<B, C, H, W>
 */
template<int B, int C, int H, int W>
class BatchNorm2d : public TorchWrapperLayer<Tensor<B, C, H, W>, Tensor<B, C, H, W>, torch::nn::BatchNorm2d> {
    using InputType = Tensor<B, C, H, W>;
    using Base = TorchWrapperLayer<InputType, InputType, torch::nn::BatchNorm2d>;
public:
    BatchNorm2d()
    : Base(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(C)))
    {}
};

/*
 * Embedding: wraps torch::nn::Embedding.
 * Maps integer indices to dense vectors.
 * Input: Tensor<B, SeqLen> (long/int indices) -> Output: Tensor<B, SeqLen, EmbedDim>
 */
template<int VocabSize, int EmbedDim>
class Embedding : public torch::nn::Module {
    torch::nn::Embedding emb;
public:
    Embedding()
    : emb(torch::nn::Embedding(torch::nn::EmbeddingOptions(VocabSize, EmbedDim)))
    {
        register_module("emb", emb);
    }

    template<int B, int SeqLen>
    Tensor<B, SeqLen, EmbedDim> forward(Tensor<B, SeqLen> input) {
        return { emb->forward(input.t()) };
    }

    // Batch-agnostic forward: BatchTensor<SeqLen> → BatchTensor<SeqLen, EmbedDim>
    template<int SeqLen>
    BatchTensor<SeqLen, EmbedDim> forward(BatchTensor<SeqLen> input) {
        return BatchTensor<SeqLen, EmbedDim>(emb->forward(input.t()));
    }
};

// Primary variadic template — specializations below for static-batch (4 params)
// and batch-agnostic (2 params).
template<int ...Dims>
class MultiHeadAttention;

/*
 * MultiHeadAttention: compile-time shape-checked multi-head attention.
 * Input/output shape: Tensor<B, SeqLen, ModelDim>
 * Template params:
 *   B        - batch size
 *   SeqLen   - sequence length
 *   NumHeads - number of attention heads
 *   ModelDim - model dimension (must be divisible by NumHeads)
 *
 * HeadDim = ModelDim / NumHeads (computed at compile time).
 * Internally projects Q, K, V via linear layers, reshapes to
 * (B, NumHeads, SeqLen, HeadDim), applies scaled dot-product attention,
 * then projects output back to ModelDim.
 */
template<int B, int SeqLen, int NumHeads, int ModelDim>
class MultiHeadAttention<B, SeqLen, NumHeads, ModelDim> : public Module<Tensor<B, SeqLen, ModelDim>, Tensor<B, SeqLen, ModelDim>> {
    static_assert(ModelDim % NumHeads == 0,
        "MultiHeadAttention: ModelDim must be divisible by NumHeads");
    static constexpr int HeadDim = ModelDim / NumHeads;

    using InputType = Tensor<B, SeqLen, ModelDim>;

    // Q, K, V projections: (B*SeqLen, ModelDim) -> (B*SeqLen, ModelDim)
    torch::nn::Linear Wq, Wk, Wv, Wo;

public:
    MultiHeadAttention()
    : Wq(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wk(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wv(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wo(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    {
        torch::nn::Module::register_module("Wq", Wq);
        torch::nn::Module::register_module("Wk", Wk);
        torch::nn::Module::register_module("Wv", Wv);
        torch::nn::Module::register_module("Wo", Wo);
    }

    InputType forward(InputType input) override {
        // input: (B, SeqLen, ModelDim)
        auto x = input.t();

        // Project Q, K, V: each (B, SeqLen, ModelDim)
        auto q = Wq->forward(x).reshape({B, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        auto k = Wk->forward(x).reshape({B, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        auto v = Wv->forward(x).reshape({B, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        // q, k, v are now (B, NumHeads, SeqLen, HeadDim)

        // Typed tensors for scaled_dot_product_attention
        using QKV = Tensor<B, NumHeads, SeqLen, HeadDim>;
        auto Q = QKV(q);
        auto K = QKV(k);
        auto V = QKV(v);

        // Scaled dot-product attention
        auto attn_out = trails::scaled_dot_product_attention(Q, K, V);
        // attn_out: (B, NumHeads, SeqLen, HeadDim)

        // Reshape back: (B, SeqLen, ModelDim)
        auto concat = attn_out.t().transpose(1, 2).contiguous().reshape({B, SeqLen, ModelDim});

        // Output projection
        return InputType(Wo->forward(concat));
    }
};

/*
 * Batch-agnostic MultiHeadAttention: BatchTensor<SeqLen, ModelDim> → BatchTensor<SeqLen, ModelDim>.
 * Works with any batch size at runtime. Forward is templated on SeqLen.
 * Template params:
 *   NumHeads - number of attention heads
 *   ModelDim - model dimension (must be divisible by NumHeads)
 */
template<int NumHeads, int ModelDim>
class MultiHeadAttention<NumHeads, ModelDim> : public torch::nn::Module {
    static_assert(ModelDim % NumHeads == 0,
        "MultiHeadAttention: ModelDim must be divisible by NumHeads");
    static constexpr int HeadDim = ModelDim / NumHeads;

    torch::nn::Linear Wq, Wk, Wv, Wo;

public:
    MultiHeadAttention()
    : Wq(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wk(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wv(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    , Wo(torch::nn::Linear(torch::nn::LinearOptions(ModelDim, ModelDim)))
    {
        register_module("Wq", Wq);
        register_module("Wk", Wk);
        register_module("Wv", Wv);
        register_module("Wo", Wo);
    }

    template<int SeqLen>
    BatchTensor<SeqLen, ModelDim> forward(BatchTensor<SeqLen, ModelDim> input) {
        auto x = input.t();  // (batch_size, SeqLen, ModelDim)
        auto batch_size = input.batch_size();

        // Project Q, K, V: each (batch_size, SeqLen, ModelDim)
        auto q = Wq->forward(x).reshape({batch_size, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        auto k = Wk->forward(x).reshape({batch_size, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        auto v = Wv->forward(x).reshape({batch_size, SeqLen, NumHeads, HeadDim}).transpose(1, 2);
        // q, k, v are now (batch_size, NumHeads, SeqLen, HeadDim)

        // Scaled dot-product attention on untyped tensors (runtime batch dim)
        const float scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
        auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
        auto weights = torch::softmax(scores, /*dim=*/-1);
        auto attn_out = torch::matmul(weights, v);
        // attn_out: (batch_size, NumHeads, SeqLen, HeadDim)

        // Reshape back: (batch_size, SeqLen, ModelDim)
        auto concat = attn_out.transpose(1, 2).contiguous().reshape({batch_size, SeqLen, ModelDim});

        // Output projection
        return BatchTensor<SeqLen, ModelDim>(Wo->forward(concat));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * RNN: wraps torch::nn::RNN with compile-time shape checking.
 * Uses batch_first=true. Input: Tensor<B, SeqLen, InputSize>
 * Returns: {output: Tensor<B, SeqLen, HiddenSize>, h_n: Tensor<NumLayers, B, HiddenSize>}
 */
template<int B, int SeqLen, int InputSize, int HiddenSize, int NumLayers = 1>
class RNN : public torch::nn::Module {
    torch::nn::RNN rnn_;
public:
    using output_t = Tensor<B, SeqLen, HiddenSize>;
    using hidden_t = Tensor<NumLayers, B, HiddenSize>;

    RNN()
    : rnn_(torch::nn::RNNOptions(InputSize, HiddenSize)
           .num_layers(NumLayers).batch_first(true))
    {
        register_module("rnn", rnn_);
    }

    std::tuple<output_t, hidden_t> forward(Tensor<B, SeqLen, InputSize> input) {
        auto [output, h_n] = rnn_->forward(input.t());
        return { output_t{output}, hidden_t{h_n} };
    }
};

/*
 * LSTM: wraps torch::nn::LSTM with compile-time shape checking.
 * Uses batch_first=true. Input: Tensor<B, SeqLen, InputSize>
 * Returns: {output: Tensor<B, SeqLen, HiddenSize>,
 *           h_n: Tensor<NumLayers, B, HiddenSize>,
 *           c_n: Tensor<NumLayers, B, HiddenSize>}
 */
template<int B, int SeqLen, int InputSize, int HiddenSize, int NumLayers = 1>
class LSTM : public torch::nn::Module {
    torch::nn::LSTM lstm_;
public:
    using output_t = Tensor<B, SeqLen, HiddenSize>;
    using hidden_t = Tensor<NumLayers, B, HiddenSize>;

    LSTM()
    : lstm_(torch::nn::LSTMOptions(InputSize, HiddenSize)
            .num_layers(NumLayers).batch_first(true))
    {
        register_module("lstm", lstm_);
    }

    std::tuple<output_t, hidden_t, hidden_t> forward(Tensor<B, SeqLen, InputSize> input) {
        auto [output, hidden_tuple] = lstm_->forward(input.t());
        auto [h_n, c_n] = hidden_tuple;
        return { output_t{output}, hidden_t{h_n}, hidden_t{c_n} };
    }
};

/*
 * GRU: wraps torch::nn::GRU with compile-time shape checking.
 * Uses batch_first=true. Input: Tensor<B, SeqLen, InputSize>
 * Returns: {output: Tensor<B, SeqLen, HiddenSize>, h_n: Tensor<NumLayers, B, HiddenSize>}
 */
template<int B, int SeqLen, int InputSize, int HiddenSize, int NumLayers = 1>
class GRU : public torch::nn::Module {
    torch::nn::GRU gru_;
public:
    using output_t = Tensor<B, SeqLen, HiddenSize>;
    using hidden_t = Tensor<NumLayers, B, HiddenSize>;

    GRU()
    : gru_(torch::nn::GRUOptions(InputSize, HiddenSize)
           .num_layers(NumLayers).batch_first(true))
    {
        register_module("gru", gru_);
    }

    std::tuple<output_t, hidden_t> forward(Tensor<B, SeqLen, InputSize> input) {
        auto [output, h_n] = gru_->forward(input.t());
        return { output_t{output}, hidden_t{h_n} };
    }
};

} // namespace trails::nn

