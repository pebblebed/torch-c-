#include <torch/torch.h>
#include <iostream>

namespace nn = torch::nn;
namespace F = torch::nn::functional;

const int B = 12;
const int D = 64;
const int H = 12;

const int layers = 17;

template <int NEmbeddings, int Dim, int NHeads, int NLayers>
class CharAttentionNet {
    nn::Embedding emb;
    nn::Sequential seq;
public:
    CharAttentionNet()
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

    auto net = CharAttentionNet<256, 128, 8, 12>();
    std::cout << net.forward("hello") << "\n";
    return 0;
}
