#include <torch/torch.h>
#include <iostream>

namespace nn = torch::nn;
namespace F = torch::nn::functional;

const int B = 12;
const int D = 64;
const int H = 12;

const int layers = 17;

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

template <int NEmbeddings, int Dim, int NHeads, int NLayers>
class CharFormer {
    nn::Embedding emb;
    nn::Sequential seq;
public:
    CharFormer()
    : emb(NEmbeddings, Dim)
    {
        for (auto i = 0; i < NLayers; i++) {
            seq->push_back(nn::TransformerEncoderLayer(Dim, NHeads));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return seq->forward(emb->forward(x));
    }

    torch::Tensor forward(std::string s) {
        std::vector<uint8_t> bytes(s.begin(), s.end());
        auto t = torch::tensor(bytes, torch::dtype(torch::kInt));
        // Add a batch dimension
        t = t.unsqueeze(0);
        return forward(t);
    }
};

int main(int argc, char** argv) {
    auto t = torch::rand({2, 3}).cuda();
    auto sq = (t * t).mean();
    std::cout << t << "\n";
    std::cout << sq << "\n";
    std::cout << t / sq << "\n";

    auto net = CharFormer<256, 128, 8, 12>();
    std::cout << net.forward("hello") << "\n";
    return 0;
}
