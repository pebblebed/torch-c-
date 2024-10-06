#include <torch/torch.h>

namespace trainium {

namespace nn = torch::nn;

template <int Dim, int NHeads>
class SelfAttention : public nn::Module {
    nn::Linear Wqkv;
    nn::MultiheadAttention attn;

public:
    SelfAttention()
    : Wqkv(Dim, 3*Dim)
    , attn(nn::MultiheadAttentionOptions().num_heads(NHeads).embed_dim(Dim)) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        assert(x.dim() == 4); // B, L, H, D
        assert(x.size(3) == Dim);
        assert(x.size(2) == NHeads);
        auto qkv = Wqkv->forward(x);
        auto q = qkv.slice(1, 0, Dim);
        auto k = qkv.slice(1, Dim, 2*Dim);
        auto v = qkv.slice(1, 2*Dim, 3*Dim);
        return attn->forward(q, k, v);
    }
};

template <int NLayers, int Dim, int OutDim=Dim, typename Activation=nn::ReLU>
class FeedForward : public nn::Module {
    nn::Sequential seq;
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

template<int Dim>
class RMSNorm : public nn::Module {
    torch::Tensor gamma;
public:
    RMSNorm()
    : gamma(register_parameter("gamma", torch::ones({Dim}))) {}

    torch::Tensor forward(torch::Tensor x) {
        return x * torch::rsqrt(x.square().mean(Dim, c10::attr::keepdim=true) + 1e-6);
    }
};

template<int Dim, typename InnerLayer, typename Norm=RMSNorm<Dim>>
class ResNorm : public nn::Module {
    InnerLayer layer;
    Norm norm;
public:
    torch::Tensor forward(torch::Tensor x) {
        return norm->forward(layer->forward(x) + x);
    }
};

template <int Dim, int NHeads, typename Activation=nn::ReLU>
class TransformerEncoderLayer : public nn::Module {
    SelfAttention<Dim, NHeads> attn;
    FeedForward<1, Dim> ff;
public:
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto a = attn->forward(x);
        auto b = ff->forward(a);
        return b;
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
    
    return pe.unsqueeze(0);  // Add batch dimension
}

torch::Tensor apply_positional_encoding(torch::Tensor x) {
    assert(x.dim() == 4);
    auto B = x.size(0);
    auto L = x.size(1);
    auto H = x.size(2);
    auto D = x.size(3);
    auto pe = positional_encoding(L, D);
    // Broadcast pe to each head
    return x + pe.unsqueeze(1).expand({B, L, H, D});
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
        // Now we need to copy out to the heads dimension
        z = z.view({x.size(0), x.size(1), NHeads, Dim});
        z = z + apply_positional_encoding(z);
        z = seq->forward(z);
        assert(z.dim() == 3); // B, L, E
        auto B = z.size(0);
        auto L = z.size(1);
        auto E = z.size(2);
        assert(E == Dim * NHeads);
        assert(B == x.size(0));
        assert(L == x.size(1));
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
}