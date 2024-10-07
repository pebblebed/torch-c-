#include <gtest/gtest.h>
#include <torch/torch.h>

#include "../charformer.hpp"

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
    auto norm = RMSNorm<64>();
    auto x = torch::randn({64});
    auto y = norm.forward(x);
    auto scale = x.square().mean().sqrt();
    auto expected = x / scale;
    auto max_err = (y - expected).abs().max().item<float>();
    // Max error is actually kinda considerable, alas
    EXPECT_LT(max_err, 1e-5);

    // Weird dims
    auto x2 = torch::randn({2, 3, 4, 7, 11});
    auto y2 = norm.forward(x2);
    EXPECT_EQ(y2.dim(), 5);
    EXPECT_EQ(y2.size(0), 2);
    EXPECT_EQ(y2.size(1), 3);
    EXPECT_EQ(y2.size(2), 4);
    EXPECT_EQ(y2.size(3), 7);
    EXPECT_EQ(y2.size(4), 11);
}

TEST(CharformerTests, Linear) {
    constexpr int InDim = 64;
    constexpr int OutDim = 32;
    Linear<InDim, OutDim> linear;
    auto x = torch::randn({InDim});
    auto y = linear.forward(x);
    EXPECT_EQ(y.dim(), 1);
    EXPECT_EQ(y.size(0), OutDim);

    // Batched
    auto x2 = torch::randn({2, InDim});
    auto y2 = linear.forward(x2);
    EXPECT_EQ(y2.dim(), 2);
    EXPECT_EQ(y2.size(0), 2);
    EXPECT_EQ(y2.size(1), OutDim);
}

TEST(CharformerTests, ResNorm) {
    constexpr int D = 64;
    ResNorm<D, Linear<D, D>> norm;
    auto x = torch::randn({D});
    auto y = norm.forward(x);
    EXPECT_EQ(y.dim(), 1);
    EXPECT_EQ(y.size(0), 64);
}
