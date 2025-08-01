#pragma once
#include <cassert>
#include <torch/torch.h>
#include "trails.hpp"

namespace trainium {

namespace nn = torch::nn;
using namespace trails;

template<
    int B,
    int ...Dims>
class RMSNorm : public Module<Tensor<B, Dims...>, Tensor<B, Dims...>> {
    Tensor<1, Dims...> gamma;
    using TensorType = Tensor<B, Dims...>;
public:
    RMSNorm()
    : gamma(torch::nn::Module::register_parameter("gamma", torch::ones({1, Dims...}))) {}

    TensorType forward(TensorType x) {
        auto variance = x.square().mean();
        auto rms = variance.rsqrt() + 1e-6;
        auto y  = x * rms;
        return y * gamma.t();
    }
};

template<
    typename InputOutput,
    template<typename, typename> class InnerLayer,
    typename Norm=RMSNorm<std::get<0>(InputOutput::size)>>
class ResNorm : public trails::Module<InputOutput, InputOutput> {
    using TensorType = InputOutput;
    InnerLayer<InputOutput, InputOutput> layer;
    Norm norm;
public:
    TensorType forward(TensorType x) {
        return norm.forward(layer.forward(x) + x);
    }
};

/*
 * Linear layer, assumes batch-first.
 */
template <class InTensor, class OutTensor>
class Linear : public trails::Module<InTensor, OutTensor> {
    nn::Linear layer;
    static_assert(2 == InTensor::dim);
    static_assert(2 == OutTensor::dim);
    constexpr static int in_dims = std::get<1>(InTensor::size);
    constexpr static int out_dims = std::get<1>(OutTensor::size);

public:
    Linear()
    : layer(in_dims, out_dims) { }

    OutTensor forward(InTensor x) {
        return layer(x);
    }
};

#if 0
template <
    int B,
    int Dim,
    int NHeads>
class SelfAttention : public nn::Module {
    using InputType = trails::Tensor<B, NHeads * Dim>;
    using WqkvType = trails::Tensor<B, 3 * NHeads * Dim>;

    Linear<InputType, WqkvType> Wqkv;
    nn::MultiheadAttention attn;

public:
    SelfAttention()
    : Wqkv()
    , attn(nn::MultiheadAttentionOptions(NHeads * Dim, NHeads)) {}

    InputType forward(InputType x) override {
        assert(x.dim() == 3); // B, L, H, D
        assert(x.size(2) == NHeads * Dim);
        // torch C++ has no batch_first option, so we need to permute
        auto qkv = Wqkv.forward(x.permute({2, 0, 1}));
        auto q = qkv.slice(1, 0, Dim);
        auto k = qkv.slice(1, Dim, 2*Dim);
        auto v = qkv.slice(1, 2*Dim, 3*Dim);
        auto pair = attn->forward(q, k, v);
        return { std::get<0>(pair) };
    }
};

template <typename InputOutput>
class ReLU : public trails::Module<InputOutput, InputOutput> {
    using TensorType = InputOutput;
public:
    TensorType forward(TensorType x) override {
        return { x.t().relu() };
    }
};

template <
    typename Activation=nn::ReLU,
    typename ...Modules>
class FeedForward : public trails::Module<B, Dim, OutDim> {
    Modules... modules;
public:
    FeedForward() {
        for (auto i = 0; i < NLayers; i++) {
            seq->push_back(nn::Linear(Dim, Dim));
            seq->push_back(Activation());
        }
        seq->push_back(nn::Linear(Dim, OutDim));
        seq->push_back(Activation());
    }

    torch::Tensor forward(torch::Tensor x) {
        return seq->forward(x);
    }
};


template <
    int B,
    int Dim,
    int NHeads,
    typename Activation=nn::ReLU>
class TransformerEncoderLayer : public nn::Module {
    ResNorm<B, Dim, SelfAttention<B, Dim, NHeads>> attn;
    ResNorm<B, Dim, FeedForward<B, 1, Dim>> ff;
public:
    torch::Tensor forward(torch::Tensor x) {
        auto a = attn->forward(x);
        auto c = ff->forward(a);
        return c;
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

template <int NEmbeddings, int Dim, int NHeads, int NLayers>
class CharFormer : public nn::Module {
    nn::Embedding emb;
    nn::Sequential seq;
    nn::Linear head;
    nn::Softmax probs;
public:
    CharFormer()
    : emb(NEmbeddings, Dim * NHeads)
    , head(Dim * NHeads, NEmbeddings)
    , probs(nn::SoftmaxOptions(1))
    {
        for (auto i = 0; i < NLayers; i++) {
            seq->push_back(nn::TransformerEncoderLayer(Dim, NHeads));
        }
        seq->push_back(head);
        seq->push_back(probs);
    }

    torch::Tensor forward(torch::Tensor x) {
        assert(x.dim() == 2); // B, L
        assert(x.dtype() == torch::kInt);
        auto z = emb->forward(x);
        auto B = z.size(0);
        auto L = z.size(1);
        auto E = z.size(2);
        assert(z.dim() == 3); // B, L, E
        // Fold out the heads dimension so positional encoding can be applied
        z = z.view({x.size(0), x.size(1), NHeads, Dim});
        z = z + apply_positional_encoding(z);
        z = seq->forward(z);
        assert(E == Dim * NHeads);
        assert(B == x.size(0));
        assert(L == x.size(1));
        z = z.view({B, L, E});
        return torch::log_softmax(head->forward(z), 2);
    }

    torch::Tensor forward(std::string s) {
        std::vector<uint8_t> bytes(s.begin(), s.end());
        auto t = torch::tensor(bytes, torch::dtype(torch::kInt));
        // Add a batch dimension
        t = t.unsqueeze(0);
        return forward(t);
    }
};
#endif

}