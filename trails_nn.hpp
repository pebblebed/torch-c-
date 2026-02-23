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

/*
 * Batch-agnostic Linear: BatchTensor<InDim> → BatchTensor<OutDim>.
 * Works with any batch size at runtime.
 */
template<int InDim, int OutDim>
class Linear : public torch::nn::Module {
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
 * Batch-agnostic LayerNorm: BatchTensor<Dim> → BatchTensor<Dim>.
 * Works with any batch size at runtime. Single normalization dimension.
 */
template<int Dim>
class LayerNorm : public torch::nn::Module {
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
 * Use LayerNorm<Dim> for single-dim normalization, or BatchLayerNorm<Dims...>
 * for multi-dim normalization.
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
 * Embedding: wraps torch::nn::Embedding.
 * Maps integer indices to dense vectors.
 * Batch-agnostic: BatchTensor<SeqLen> (long/int indices) -> BatchTensor<SeqLen, EmbedDim>
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

    // Batch-agnostic forward: BatchTensor<SeqLen> → BatchTensor<SeqLen, EmbedDim>
    template<int SeqLen>
    BatchTensor<SeqLen, EmbedDim> forward(BatchTensor<SeqLen> input) {
        return BatchTensor<SeqLen, EmbedDim>(emb->forward(input.t()));
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
class MultiHeadAttention : public torch::nn::Module {
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
 * Batch-agnostic BatchNorm1d: BatchTensor<C, L> → BatchTensor<C, L>.
 * Works with any batch size at runtime. Forward is templated on L.
 * Template param:
 *   C - number of channels (features)
 */
template<int C>
class BatchNorm1d : public torch::nn::Module {
    torch::nn::BatchNorm1d inner_;
public:
    BatchNorm1d()
    : inner_(torch::nn::BatchNorm1d(torch::nn::BatchNormOptions(C)))
    {
        register_module("batch_norm1d", inner_);
    }

    template<int L>
    BatchTensor<C, L> forward(BatchTensor<C, L> input) {
        return BatchTensor<C, L>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * Batch-agnostic BatchNorm2d: BatchTensor<C, H, W> → BatchTensor<C, H, W>.
 * Works with any batch size at runtime. Forward is templated on H, W.
 * Template param:
 *   C - number of channels (features)
 */
template<int C>
class BatchNorm2d : public torch::nn::Module {
    torch::nn::BatchNorm2d inner_;
public:
    BatchNorm2d()
    : inner_(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(C)))
    {
        register_module("batch_norm2d", inner_);
    }

    template<int H, int W>
    BatchTensor<C, H, W> forward(BatchTensor<C, H, W> input) {
        return BatchTensor<C, H, W>(inner_->forward(input.t()));
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * Batch-agnostic RNN: wraps torch::nn::RNN with batch_first=true.
 * Forward takes BatchTensor<SeqLen, InputSize> (SeqLen templated).
 * Returns {output: BatchTensor<SeqLen, HiddenSize>, h_n: torch::Tensor}.
 * h_n has shape (NumLayers, B, HiddenSize) which doesn't fit BatchTensor
 * cleanly, so it is returned as a raw torch::Tensor.
 */
template<int InputSize, int HiddenSize, int NumLayers = 1>
class RNN : public torch::nn::Module {
    torch::nn::RNN rnn_;
public:
    RNN()
    : rnn_(torch::nn::RNNOptions(InputSize, HiddenSize)
           .num_layers(NumLayers).batch_first(true))
    {
        register_module("rnn", rnn_);
    }

    template<int SeqLen>
    std::tuple<BatchTensor<SeqLen, HiddenSize>, torch::Tensor>
    forward(BatchTensor<SeqLen, InputSize> input) {
        auto [output, h_n] = rnn_->forward(input.t());
        return { BatchTensor<SeqLen, HiddenSize>{output}, h_n };
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * Batch-agnostic LSTM: wraps torch::nn::LSTM with batch_first=true.
 * Forward takes BatchTensor<SeqLen, InputSize> (SeqLen templated).
 * Returns {output: BatchTensor<SeqLen, HiddenSize>, h_n: torch::Tensor, c_n: torch::Tensor}.
 * h_n and c_n have shape (NumLayers, B, HiddenSize) — returned as raw torch::Tensor.
 */
template<int InputSize, int HiddenSize, int NumLayers = 1>
class LSTM : public torch::nn::Module {
    torch::nn::LSTM lstm_;
public:
    LSTM()
    : lstm_(torch::nn::LSTMOptions(InputSize, HiddenSize)
            .num_layers(NumLayers).batch_first(true))
    {
        register_module("lstm", lstm_);
    }

    template<int SeqLen>
    std::tuple<BatchTensor<SeqLen, HiddenSize>, torch::Tensor, torch::Tensor>
    forward(BatchTensor<SeqLen, InputSize> input) {
        auto [output, hidden_tuple] = lstm_->forward(input.t());
        auto [h_n, c_n] = hidden_tuple;
        return { BatchTensor<SeqLen, HiddenSize>{output}, h_n, c_n };
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

/*
 * Batch-agnostic GRU: wraps torch::nn::GRU with batch_first=true.
 * Forward takes BatchTensor<SeqLen, InputSize> (SeqLen templated).
 * Returns {output: BatchTensor<SeqLen, HiddenSize>, h_n: torch::Tensor}.
 * h_n has shape (NumLayers, B, HiddenSize) — returned as raw torch::Tensor.
 */
template<int InputSize, int HiddenSize, int NumLayers = 1>
class GRU : public torch::nn::Module {
    torch::nn::GRU gru_;
public:
    GRU()
    : gru_(torch::nn::GRUOptions(InputSize, HiddenSize)
           .num_layers(NumLayers).batch_first(true))
    {
        register_module("gru", gru_);
    }

    template<int SeqLen>
    std::tuple<BatchTensor<SeqLen, HiddenSize>, torch::Tensor>
    forward(BatchTensor<SeqLen, InputSize> input) {
        auto [output, h_n] = gru_->forward(input.t());
        return { BatchTensor<SeqLen, HiddenSize>{output}, h_n };
    }

    std::vector<torch::Tensor> parameters() const {
        return torch::nn::Module::parameters();
    }
};

} // namespace trails::nn

