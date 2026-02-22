#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "../charformer.hpp"
#include "../trails_nn.hpp"

using namespace trainium;

TEST(CharformerTests, PosEncoding) {
    auto enc = positional_encoding(15, 128);
    EXPECT_EQ(enc.dim(),  2);
    EXPECT_EQ(enc.size(0), 15);
    EXPECT_EQ(enc.size(1), 128);
}

TEST(CharformerTests, ApplyPosEncoding) {
    const auto sizes = std::tuple{11, 1024, 12, 64};
    const auto [B, L, H, D] = sizes;
    auto x = torch::randn({B, L, H, D});
    auto y = apply_positional_encoding(x);
    EXPECT_EQ(y.dim(), 4);
    EXPECT_EQ(y.size(0), B);
    EXPECT_EQ(y.size(1), L);
    EXPECT_EQ(y.size(2), H);
    EXPECT_EQ(y.size(3), D);
}

TEST(CharformerTests, RMSNorm) {
    constexpr int B = 1;
    constexpr int D = 3;
    auto norm = RMSNorm<B, D>();
    auto x = trails::Tensor<B, D>::randn();
    auto y = norm.forward(x);
    auto scale = ::sqrt(x.square().mean().item<float>());
    auto expected = x / scale;
    auto max_err = (y - expected).abs().max().item<float>();
    std::cerr << "x: " << x << "; scale: " << scale << std::endl;
    std::cerr << "y: " << y << std::endl;
    std::cerr << "expected: " << expected << std::endl;
    std::cerr << "diffs: " << (y - expected) << std::endl;
    // Max error is actually kinda considerable, alas
    EXPECT_LT(max_err, 1e-5f);

    // Weird dims
    auto newNorm = RMSNorm<2, 3, 4, 7, 11>();
    auto x2 = torch::randn({2, 3, 4, 7, 11});
    auto y2 = newNorm.forward(x2);
    EXPECT_EQ(y2.dim(), 5);
    EXPECT_EQ(y2.size<0>, 2);
    EXPECT_EQ(y2.size<1>, 3);
    EXPECT_EQ(y2.size<2>, 4);
    EXPECT_EQ(y2.size<3>, 7);
    EXPECT_EQ(y2.size<4>, 11);
}

TEST(CharformerTests, Linear) {
    constexpr int B = 1;
    constexpr int InDim = 64;
    constexpr int OutDim = 32;
    trails::nn::Linear<B, InDim, OutDim> linear;
    auto x = torch::randn({B, InDim});
    auto y = linear.forward(x);
    EXPECT_EQ(y.dim(), 2);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, OutDim);
}

TEST(CharformerTests, ResNorm) {
    // ResNorm<TensorType, InnerLayerTemplate, Norm> applies:
    //   norm(layer(x) + x)
    // where InnerLayer is a template<typename In, typename Out> class.
    // trainium::Linear matches this signature.
    constexpr int B = 1;
    constexpr int D = 64;
    using TensorType = trails::Tensor<B, D>;
    ResNorm<TensorType, trainium::Linear, RMSNorm<B, D>> norm;
    auto x = TensorType::randn();
    auto y = norm.forward(x);
    EXPECT_EQ(y.dim(), 2);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, D);
}

TEST(CharformerTests, SelfAttention) {
    // Old SelfAttention class is #if 0'd. Use trails::nn::MultiHeadAttention instead.
    constexpr int B = 3;
    constexpr int L = 16;  // reduced from 128 for test speed
    constexpr int H = 10;
    constexpr int D = 640; // ModelDim = NumHeads * HeadDim = 10 * 64
    trails::nn::MultiHeadAttention<B, L, H, D> attn;
    auto x = trails::Tensor<B, L, D>::randn();
    auto y = attn.forward(x);
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, L);
    EXPECT_EQ(y.size<2>, D);
}

TEST(CharformerTests, MultiHeadAttention) {
    // B=2, SeqLen=16, NumHeads=4, ModelDim=64 (HeadDim=16)
    constexpr int B = 2;
    constexpr int L = 16;
    constexpr int H = 4;
    constexpr int D = 64;
    trails::nn::MultiHeadAttention<B, L, H, D> mha;
    auto x = trails::Tensor<B, L, D>::randn();
    auto y = mha.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{B, L, D}));
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, L);
    EXPECT_EQ(y.size<2>, D);
}


// ============================================================
// FeedForward tests
// ============================================================

TEST(CharformerTests, FeedForwardShape) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, ModelDim = 32, FFDim = 64;
    FeedForward<B, SeqLen, ModelDim, FFDim> ff;
    auto x = trails::Tensor<B, SeqLen, ModelDim>::randn();
    auto y = ff.forward(x);
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, SeqLen);
    EXPECT_EQ(y.size<2>, ModelDim);
}

TEST(CharformerTests, FeedForwardOutputFinite) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, ModelDim = 32, FFDim = 64;
    FeedForward<B, SeqLen, ModelDim, FFDim> ff;
    auto x = trails::Tensor<B, SeqLen, ModelDim>::randn();
    auto y = ff.forward(x);
    // Output should be finite (no NaN or Inf)
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());
    // Output should differ from input (not identity)
    auto diff = (y.t() - x.t()).abs().sum().item<float>();
    EXPECT_GT(diff, 0.0f);
}

// ============================================================
// TransformerEncoderLayer tests
// ============================================================

TEST(CharformerTests, TransformerEncoderLayerShape) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, NumHeads = 2, ModelDim = 32, FFDim = 64;
    TransformerEncoderLayer<B, SeqLen, NumHeads, ModelDim, FFDim> layer;
    auto x = trails::Tensor<B, SeqLen, ModelDim>::randn();
    auto y = layer.forward(x);
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, SeqLen);
    EXPECT_EQ(y.size<2>, ModelDim);
}

TEST(CharformerTests, TransformerEncoderLayerOutputFinite) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, NumHeads = 2, ModelDim = 32, FFDim = 64;
    TransformerEncoderLayer<B, SeqLen, NumHeads, ModelDim, FFDim> layer;
    auto x = trails::Tensor<B, SeqLen, ModelDim>::randn();
    auto y = layer.forward(x);
    // Output should be finite (no NaN or Inf)
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());
}

// ============================================================
// CharFormer tests
// ============================================================

TEST(CharformerTests, CharFormerForward) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, VocabSize = 256, ModelDim = 32, NumHeads = 2, FFDim = 64, NLayers = 2;
    CharFormer<B, SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers> model;
    // Input: random long indices in [0, VocabSize)
    auto x = trails::Tensor<B, SeqLen>(torch::randint(0, VocabSize, {B, SeqLen}, torch::kLong));
    auto y = model.forward(x);

    // Check output shape: (B, SeqLen, VocabSize)
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, SeqLen);
    EXPECT_EQ(y.size<2>, VocabSize);

    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());

    // All values should be <= 0 (log probabilities)
    EXPECT_TRUE(torch::all(y.t() <= 0.0f).item<bool>());

    // exp along last dim should sum to ~1 (valid probability distribution)
    auto probs = y.t().exp().sum(/*dim=*/-1);
    auto max_err = (probs - 1.0f).abs().max().item<float>();
    EXPECT_LT(max_err, 1e-4f);
}

TEST(CharformerTests, CharFormerStringForward) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, VocabSize = 256, ModelDim = 32, NumHeads = 2, FFDim = 64, NLayers = 2;
    CharFormer<B, SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers> model;
    auto y = model.forward("test");

    // Check output shape: (B, SeqLen, VocabSize)
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, SeqLen);
    EXPECT_EQ(y.size<2>, VocabSize);

    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());

    // All values should be <= 0 (log probabilities)
    EXPECT_TRUE(torch::all(y.t() <= 0.0f).item<bool>());
}
