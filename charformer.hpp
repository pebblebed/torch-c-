#pragma once
#include <array>
#include <cassert>
#include <string>
#include <torch/torch.h>
#include "trails/trails.hpp"
#include "trails/trails_nn.hpp"

namespace trainium {

namespace nn = torch::nn;
using namespace trails;
using namespace trails::nn;

template<int ...Dims>
class RMSNorm : public torch::nn::Module {
    torch::Tensor gamma_;
    using TensorType = BatchTensor<Dims...>;
public:
    RMSNorm()
    : gamma_(torch::nn::Module::register_parameter("gamma", torch::ones({Dims...}))) {}

    TensorType forward(TensorType x) {
        auto xt = x.t();
        // Reduce over all non-batch dimensions (1..sizeof...(Dims)), keepdim for broadcasting
        std::vector<int64_t> reduce_dims;
        for (int64_t i = 1; i <= (int64_t)sizeof...(Dims); i++) {
            reduce_dims.push_back(i);
        }
        auto variance = (xt * xt).mean(reduce_dims, /*keepdim=*/true);
        auto rms = (variance + 1e-6).rsqrt();
        return TensorType(xt * rms * gamma_);
    }
};

/*
 * ResNorm: residual + normalization wrapper.
 * Applies: norm(layer(x) + x)
 * Template params:
 *   Dims... - the mathematical dimensions of the BatchTensor
 *   Layer   - a module type whose forward takes and returns BatchTensor<Dims...>
 *   Norm    - a normalization module (defaults to RMSNorm<Dims...>)
 */
template<typename Layer, typename Norm>
class ResNorm : public torch::nn::Module {
    Layer layer;
    Norm norm;
public:
    template<typename T>
    T forward(T x) {
        return norm.forward(layer.forward(x) + x);
    }
};

/*
 * FeedForward: two-layer MLP with ReLU activation.
 * ModelDim -> FFDim -> ModelDim
 * Batch-agnostic: works with any batch size at runtime.
 */
template<int SeqLen, int ModelDim, int FFDim>
class FeedForward : public torch::nn::Module {
    using InputType = BatchTensor<SeqLen, ModelDim>;

    Tensor<FFDim, ModelDim> w1;
    Tensor<FFDim> b1;
    Tensor<ModelDim, FFDim> w2;
    Tensor<ModelDim> b2;

public:
    FeedForward()
    : w1(torch::nn::Module::register_parameter("w1", Tensor<FFDim, ModelDim>::randn().t()))
    , b1(torch::nn::Module::register_parameter("b1", Tensor<FFDim>::randn().t()))
    , w2(torch::nn::Module::register_parameter("w2", Tensor<ModelDim, FFDim>::randn().t()))
    , b2(torch::nn::Module::register_parameter("b2", Tensor<ModelDim>::randn().t()))
    {}

    InputType forward(InputType x) {
        auto h = trails::functional::linear(x, w1, std::optional{b1});
        h = trails::functional::relu(h);
        return trails::functional::linear(h, w2, std::optional{b2});
    }
};

/*
 * TransformerEncoderLayer: MHA + residual + LayerNorm, then FF + residual + LayerNorm.
 * Batch-agnostic: works with any batch size at runtime.
 * Input/output: BatchTensor<SeqLen, ModelDim>
 */
template<int SeqLen, int NumHeads, int ModelDim, int FFDim>
class TransformerEncoderLayer : public torch::nn::Module {
public:
    using InputType = BatchTensor<SeqLen, ModelDim>;

    std::shared_ptr<trails::nn::MultiHeadAttention<NumHeads, ModelDim>> mha;
    std::shared_ptr<trails::nn::BatchLayerNorm<SeqLen, ModelDim>> ln1;
    std::shared_ptr<FeedForward<SeqLen, ModelDim, FFDim>> ff;
    std::shared_ptr<trails::nn::BatchLayerNorm<SeqLen, ModelDim>> ln2;

    TransformerEncoderLayer()
    : mha(std::make_shared<trails::nn::MultiHeadAttention<NumHeads, ModelDim>>())
    , ln1(std::make_shared<trails::nn::BatchLayerNorm<SeqLen, ModelDim>>())
    , ff(std::make_shared<FeedForward<SeqLen, ModelDim, FFDim>>())
    , ln2(std::make_shared<trails::nn::BatchLayerNorm<SeqLen, ModelDim>>())
    {
        register_module("mha", mha);
        register_module("ln1", ln1);
        register_module("ff", ff);
        register_module("ln2", ln2);
    }

    InputType forward(InputType x) {
        // Self-attention sublayer with residual + LayerNorm
        auto attn_out = mha->template forward<SeqLen>(x);
        auto x1 = ln1->forward(attn_out + x);
        // Feedforward sublayer with residual + LayerNorm
        auto ff_out = ff->forward(x1);
        return ln2->forward(ff_out + x1);
    }
};

using namespace torch::indexing;

torch::Tensor positional_encoding(int64_t seq_length, int64_t hidden_dim) {
    auto position = torch::arange(0, seq_length, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, hidden_dim, 2, torch::kFloat32) *
                               (-std::log(10000.0) / hidden_dim));

    auto pe = torch::zeros({seq_length, hidden_dim});
    pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
    pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));
    assert(pe.dim() == 2);
    assert(pe.size(0) == seq_length);
    assert(pe.size(1) == hidden_dim);
    return pe;
}

torch::Tensor apply_positional_encoding(torch::Tensor x) {
    assert(x.dim() == 4);
    auto B = x.size(0);
    auto L = x.size(1);
    auto H = x.size(2);
    auto D = x.size(3);
    auto pe = positional_encoding(L, D);
    // pe is now L, D. Pop on singleton B, H dimensions at beginning and end,
    // and permute H into place for expand()
    pe = pe.unsqueeze(0).unsqueeze(-1).permute({0, 1, 3, 2}).expand({B, L, H, D});
    return x + pe;
}

/*
 * CharFormer: character-level transformer language model.
 * Embedding + sinusoidal positional encoding + NLayers encoder layers + linear head + log_softmax.
 * Batch-agnostic: works with any batch size at runtime.
 * Input: BatchTensor<SeqLen> of int64 indices -> Output: BatchTensor<SeqLen, VocabSize> log probabilities.
 */
template<int SeqLen, int VocabSize, int ModelDim, int NumHeads, int FFDim, int NLayers>
class CharFormer : public torch::nn::Module {
    using InputType = BatchTensor<SeqLen>;
    using OutputType = BatchTensor<SeqLen, VocabSize>;

    std::shared_ptr<trails::nn::Embedding<VocabSize, ModelDim>> emb;
    std::array<std::shared_ptr<TransformerEncoderLayer<SeqLen, NumHeads, ModelDim, FFDim>>, NLayers> layers;
    Tensor<VocabSize, ModelDim> head_w;
    Tensor<VocabSize> head_b;

public:
    CharFormer()
    : emb(std::make_shared<trails::nn::Embedding<VocabSize, ModelDim>>())
    , head_w(torch::nn::Module::register_parameter("head_w", Tensor<VocabSize, ModelDim>::randn().t()))
    , head_b(torch::nn::Module::register_parameter("head_b", Tensor<VocabSize>::randn().t()))
    {
        register_module("emb", emb);
        for (int i = 0; i < NLayers; i++) {
            layers[i] = std::make_shared<TransformerEncoderLayer<SeqLen, NumHeads, ModelDim, FFDim>>();
            register_module("layer_" + std::to_string(i), layers[i]);
        }
    }

    OutputType forward(InputType x) {
        // Embedding: BatchTensor<SeqLen> -> BatchTensor<SeqLen, ModelDim>
        auto z = emb->template forward<SeqLen>(x);

        // Add sinusoidal positional encoding
        // pe is (SeqLen, ModelDim), unsqueeze(0) -> (1, SeqLen, ModelDim), broadcasts over batch
        auto pe = positional_encoding(SeqLen, ModelDim);
        z = BatchTensor<SeqLen, ModelDim>(z.t() + pe.unsqueeze(0));

        // Encoder layers
        for (int i = 0; i < NLayers; i++) {
            z = layers[i]->forward(z);
        }

        // Linear head + log_softmax: BatchTensor<SeqLen, ModelDim> -> BatchTensor<SeqLen, VocabSize>
        auto logits = trails::functional::linear(z, head_w, std::optional{head_b});
        return logits.template log_softmax<1>();
    }

    OutputType forward(std::string s) {
        std::vector<int64_t> bytes(s.begin(), s.end());
        auto t = torch::tensor(bytes, torch::dtype(torch::kLong));
        // Add a batch dimension and pad/truncate to SeqLen
        t = t.unsqueeze(0);
        // Ensure correct sequence length
        if (t.size(1) < SeqLen) {
            t = torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, SeqLen - t.size(1)}));
        } else if (t.size(1) > SeqLen) {
            t = t.slice(1, 0, SeqLen);
        }
        // t is now (1, SeqLen) — a single-element batch
        return forward(InputType(t));
    }
};

}