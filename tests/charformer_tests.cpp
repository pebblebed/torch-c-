#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "../charformer.hpp"
#include "trails/trails_nn.hpp"

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
    auto norm = RMSNorm<D>();
    auto x = trails::BatchTensor<D>(torch::randn({B, D}));
    auto y = norm.forward(x);
    auto scale = ::sqrt((x.t() * x.t()).mean().item<float>() + 1e-6f);
    auto expected_t = x.t() / scale;
    auto max_err = (y.t() - expected_t).abs().max().item<float>();
    std::cerr << "x: " << x << "; scale: " << scale << std::endl;
    std::cerr << "y: " << y << std::endl;
    std::cerr << "expected: " << expected_t << std::endl;
    std::cerr << "diffs: " << (y.t() - expected_t) << std::endl;
    // Max error is actually kinda considerable, alas
    EXPECT_LT(max_err, 1e-5f);

    // Weird dims (batch-agnostic: no B in template)
    auto newNorm = RMSNorm<3, 4, 7, 11>();
    auto x2 = trails::BatchTensor<3, 4, 7, 11>(torch::randn({2, 3, 4, 7, 11}));
    auto y2 = newNorm.forward(x2);
    EXPECT_EQ(y2.t().dim(), 5);
    EXPECT_EQ(y2.batch_size(), 2);
    EXPECT_EQ(y2.t().size(1), 3);
    EXPECT_EQ(y2.t().size(2), 4);
    EXPECT_EQ(y2.t().size(3), 7);
    EXPECT_EQ(y2.t().size(4), 11);
}

TEST(CharformerTests, Linear) {
    constexpr int B = 1;
    constexpr int InDim = 64;
    constexpr int OutDim = 32;
    trails::nn::Linear<InDim, OutDim> linear;
    auto x = trails::BatchTensor<InDim>(torch::randn({B, InDim}));
    auto y = linear.forward(x);
    EXPECT_EQ(y.t().dim(), 2);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), OutDim);
}

TEST(CharformerTests, ResNorm) {
    // ResNorm<Layer, Norm> applies: norm(layer(x) + x)
    // Layer and Norm are concrete types that support forward().
    constexpr int B = 1;
    constexpr int D = 64;
    ResNorm<trails::nn::Linear<D, D>, RMSNorm<D>> norm;
    auto x = trails::BatchTensor<D>(torch::randn({B, D}));
    auto y = norm.forward(x);
    EXPECT_EQ(y.t().dim(), 2);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), D);
}

TEST(CharformerTests, SelfAttention) {
    // Use batch-agnostic trails::nn::MultiHeadAttention<NumHeads, ModelDim>
    constexpr int B = 3;
    constexpr int L = 16;  // reduced from 128 for test speed
    constexpr int H = 10;
    constexpr int D = 640; // ModelDim = NumHeads * HeadDim = 10 * 64
    trails::nn::MultiHeadAttention<H, D> attn;
    auto x = trails::BatchTensor<L, D>(torch::randn({B, L, D}));
    auto y = attn.forward<L>(x);
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), L);
    EXPECT_EQ(y.t().size(2), D);
}

TEST(CharformerTests, MultiHeadAttention) {
    // B=2, SeqLen=16, NumHeads=4, ModelDim=64 (HeadDim=16)
    constexpr int B = 2;
    constexpr int L = 16;
    constexpr int H = 4;
    constexpr int D = 64;
    trails::nn::MultiHeadAttention<H, D> mha;
    auto x = trails::BatchTensor<L, D>(torch::randn({B, L, D}));
    auto y = mha.forward<L>(x);
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), L);
    EXPECT_EQ(y.t().size(2), D);
}


// ============================================================
// FeedForward tests
// ============================================================

TEST(CharformerTests, FeedForwardShape) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, ModelDim = 32, FFDim = 64;
    FeedForward<SeqLen, ModelDim, FFDim> ff;
    auto x = trails::BatchTensor<SeqLen, ModelDim>(torch::randn({B, SeqLen, ModelDim}));
    auto y = ff.forward(x);
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), SeqLen);
    EXPECT_EQ(y.t().size(2), ModelDim);
}

TEST(CharformerTests, FeedForwardOutputFinite) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, ModelDim = 32, FFDim = 64;
    FeedForward<SeqLen, ModelDim, FFDim> ff;
    auto x = trails::BatchTensor<SeqLen, ModelDim>(torch::randn({B, SeqLen, ModelDim}));
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
    TransformerEncoderLayer<SeqLen, NumHeads, ModelDim, FFDim> layer;
    auto x = trails::BatchTensor<SeqLen, ModelDim>(torch::randn({B, SeqLen, ModelDim}));
    auto y = layer.forward(x);
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), SeqLen);
    EXPECT_EQ(y.t().size(2), ModelDim);
}

TEST(CharformerTests, TransformerEncoderLayerOutputFinite) {
    torch::NoGradGuard no_grad;
    constexpr int B = 2, SeqLen = 8, NumHeads = 2, ModelDim = 32, FFDim = 64;
    TransformerEncoderLayer<SeqLen, NumHeads, ModelDim, FFDim> layer;
    auto x = trails::BatchTensor<SeqLen, ModelDim>(torch::randn({B, SeqLen, ModelDim}));
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
    CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers> model;
    // Input: random long indices in [0, VocabSize)
    auto x = trails::BatchTensor<SeqLen>(torch::randint(0, VocabSize, {B, SeqLen}, torch::kLong));
    auto y = model.forward(x);

    // Check output shape: (B, SeqLen, VocabSize)
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), B);
    EXPECT_EQ(y.t().size(1), SeqLen);
    EXPECT_EQ(y.t().size(2), VocabSize);

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
    constexpr int SeqLen = 8, VocabSize = 256, ModelDim = 32, NumHeads = 2, FFDim = 64, NLayers = 2;
    CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers> model;
    auto y = model.forward("test");

    // Check output shape: (1, SeqLen, VocabSize) — single-element batch
    EXPECT_EQ(y.t().dim(), 3);
    EXPECT_EQ(y.batch_size(), 1);
    EXPECT_EQ(y.t().size(1), SeqLen);
    EXPECT_EQ(y.t().size(2), VocabSize);

    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());

    // All values should be <= 0 (log probabilities)
    EXPECT_TRUE(torch::all(y.t() <= 0.0f).item<bool>());
}

// ============================================================
// Wave 3, Task 6: Batch-agnostic CharFormer module tests
// TODO: uncomment after Task 5 completes (BatchTensor-ize charformer.hpp)
// These tests require FeedForward, TransformerEncoderLayer, and CharFormer
// to accept BatchTensor inputs without a compile-time B parameter.
// ============================================================

TEST(CharformerBatchTests, FeedForward_Batch) {
    torch::NoGradGuard no_grad;
    // FeedForward<SeqLen=8, ModelDim=32, FFDim=64> (no B!)
    // Forward with BatchTensor<8, 32> batch_size=3
    FeedForward<8, 32, 64> ff;
    auto input = trails::BatchTensor<8, 32>(torch::randn({3, 8, 32}));
    auto output = ff.forward(input);
    ASSERT_EQ(output.batch_size(), 3);
    ASSERT_EQ(output.t().size(1), 8);
    ASSERT_EQ(output.t().size(2), 32);
    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(output.t())).item<bool>());
}

TEST(CharformerBatchTests, TransformerEncoderLayer_Batch) {
    torch::NoGradGuard no_grad;
    // TransformerEncoderLayer<SeqLen=8, NumHeads=2, ModelDim=32, FFDim=64> (no B!)
    TransformerEncoderLayer<8, 2, 32, 64> layer;
    auto input = trails::BatchTensor<8, 32>(torch::randn({3, 8, 32}));
    auto output = layer.forward(input);
    ASSERT_EQ(output.batch_size(), 3);
    ASSERT_EQ(output.t().size(1), 8);
    ASSERT_EQ(output.t().size(2), 32);
    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(output.t())).item<bool>());
}

TEST(CharformerBatchTests, CharFormer_BatchDynamic) {
    torch::NoGradGuard no_grad;
    // CharFormer<SeqLen=8, VocabSize=256, ModelDim=32, NumHeads=2, FFDim=64, NLayers=2> (no B!)
    // KEY BENEFIT: same model, different batch sizes at runtime
    CharFormer<8, 256, 32, 2, 64, 2> model;

    // Forward with batch_size=1
    auto input1 = trails::BatchTensor<8>(torch::randint(0, 256, {1, 8}, torch::kLong));
    auto output1 = model.forward(input1);
    ASSERT_EQ(output1.batch_size(), 1);
    ASSERT_EQ(output1.t().size(1), 8);
    ASSERT_EQ(output1.t().size(2), 256);
    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(output1.t())).item<bool>());
    // All values should be <= 0 (log probabilities)
    EXPECT_TRUE(torch::all(output1.t() <= 0.0f).item<bool>());
    // exp along last dim should sum to ~1
    auto probs1 = output1.t().exp().sum(-1);
    EXPECT_NEAR(probs1.mean().item<float>(), 1.0f, 1e-4f);

    // Forward with batch_size=4 — same model!
    auto input4 = trails::BatchTensor<8>(torch::randint(0, 256, {4, 8}, torch::kLong));
    auto output4 = model.forward(input4);
    ASSERT_EQ(output4.batch_size(), 4);
    ASSERT_EQ(output4.t().size(1), 8);
    ASSERT_EQ(output4.t().size(2), 256);
    // Output should be finite
    EXPECT_TRUE(torch::all(torch::isfinite(output4.t())).item<bool>());
    // All values should be <= 0 (log probabilities)
    EXPECT_TRUE(torch::all(output4.t() <= 0.0f).item<bool>());
    // exp along last dim should sum to ~1
    auto probs4 = output4.t().exp().sum(-1);
    auto max_err = (probs4 - 1.0f).abs().max().item<float>();
    EXPECT_LT(max_err, 1e-4f);
}

TEST(CharformerTests, LanguageModelLossMatchesNllGradientOnLogProbs) {
    constexpr int B = 3;
    constexpr int SeqLen = 7;
    constexpr int Vocab = 11;
    auto raw_logits = torch::randn({B, SeqLen, Vocab});
    auto log_probs = torch::log_softmax(raw_logits, /*dim=*/-1);
    auto labels = torch::randint(0, Vocab, {B, SeqLen}, torch::kLong);

    auto input_a = log_probs.detach().clone().set_requires_grad(true);
    auto input_b = log_probs.detach().clone().set_requires_grad(true);

    auto got = language_model_loss(input_a, labels);
    auto expected = torch::nn::functional::nll_loss(
        input_b.reshape({B * SeqLen, Vocab}),
        labels.reshape({B * SeqLen}));

    got.backward();
    expected.backward();

    EXPECT_TRUE(torch::allclose(
        input_a.grad(),
        input_b.grad(),
        1e-6,
        1e-6));
}

// ============================================================
// CUDA tests for cuda() method
// ============================================================

TEST(CudaTests, LinearCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    trails::nn::Linear<32, 16> linear;
    linear.cuda();
    for (const auto& p : linear.parameters()) {
        EXPECT_TRUE(p.is_cuda());
    }
}

TEST(CudaTests, CharFormerCuda) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    CharFormer<8, 256, 32, 2, 64, 2> model;
    model.cuda();
    for (const auto& p : model.parameters()) {
        EXPECT_TRUE(p.is_cuda()) << "Parameter not on CUDA: " << p.sizes();
    }
}

TEST(CudaTests, CharFormerCudaForward) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    torch::NoGradGuard no_grad;
    CharFormer<8, 256, 32, 2, 64, 2> model;
    model.cuda();
    auto x = trails::BatchTensor<8>(
        torch::randint(0, 256, {2, 8}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)));
    auto y = model.forward(x);
    EXPECT_TRUE(y.t().is_cuda());
    EXPECT_EQ(y.batch_size(), 2);
    EXPECT_EQ(y.t().size(1), 8);
    EXPECT_EQ(y.t().size(2), 256);
    EXPECT_TRUE(torch::all(torch::isfinite(y.t())).item<bool>());
    EXPECT_TRUE(torch::all(y.t() <= 0.0f).item<bool>());
}

// ============================================================
// MPS tests for mps() method
// ============================================================

TEST(MpsTests, LinearMps) {
    if (!torch::mps::is_available()) {
        GTEST_SKIP() << "MPS not available";
    }
    trails::nn::Linear<32, 16> linear;
    linear.mps();
    for (const auto& p : linear.parameters()) {
        EXPECT_TRUE(p.is_mps());
    }
}

TEST(MpsTests, CharFormerMps) {
    if (!torch::mps::is_available()) {
        GTEST_SKIP() << "MPS not available";
    }
    CharFormer<8, 256, 32, 2, 64, 2> model;
    model.mps();
    for (const auto& p : model.parameters()) {
        EXPECT_TRUE(p.is_mps()) << "Parameter not on MPS: " << p.sizes();
    }
}

TEST(MpsTests, CharFormerMpsForward) {
    if (!torch::mps::is_available()) {
        GTEST_SKIP() << "MPS not available";
    }
    torch::NoGradGuard no_grad;
    CharFormer<8, 256, 32, 2, 64, 2> model;
    model.mps();
    auto x = trails::BatchTensor<8>(
        torch::randint(0, 256, {2, 8}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMPS)));
    auto y = model.forward(x);
    EXPECT_TRUE(y.t().is_mps());
    EXPECT_EQ(y.batch_size(), 2);
    EXPECT_EQ(y.t().size(1), 8);
    EXPECT_EQ(y.t().size(2), 256);
    EXPECT_TRUE(torch::all(torch::isfinite(y.t().to(torch::kCPU))).item<bool>());
    EXPECT_TRUE(torch::all(y.t().to(torch::kCPU) <= 0.0f).item<bool>());
}
