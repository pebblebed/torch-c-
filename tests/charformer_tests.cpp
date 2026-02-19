#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "../charformer.hpp"

using namespace trainium;

#if 0
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

#endif

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
    trails::Linear<B, InDim, OutDim> linear;
    auto x = torch::randn({B, InDim});
    auto y = linear.forward(x);
    EXPECT_EQ(y.dim(), 2);
    EXPECT_EQ(y.size<0>, B);
    EXPECT_EQ(y.size<1>, OutDim);
}

TEST(CharformerTests, ResNorm) {
#if 0
    constexpr int B = 1;
    constexpr int D = 64;
    using TensorType = Tensor<B, D>;
    ResNorm<TensorType, Linear> norm;
    auto x = torch::randn({D});
    auto y = norm.forward(x);
    EXPECT_EQ(y.dim(), 1);
    EXPECT_EQ(y.size(0), 64);

    // Weird dims
    auto x2 = torch::randn({2, 3, 4, 7, 11});
    constexpr int D2 = 2 * 3 * 4 * 7 * 11;
    ResNorm<D2, RMSNorm<D2>> norm2;
    auto y2 = norm2.forward(x2);
    EXPECT_EQ(y2.dim(), 5);
    EXPECT_EQ(y2.size(0), 2);
    EXPECT_EQ(y2.size(1), 3);
    EXPECT_EQ(y2.size(2), 4);
    EXPECT_EQ(y2.size(3), 7);
    EXPECT_EQ(y2.size(4), 11);
#endif
}

TEST(CharformerTests, SelfAttention) {
#if 0
    constexpr int B = 3;
    constexpr int L = 128;
    constexpr int H = 10;
    constexpr int D = 64;
    SelfAttention<D, H> attn;
    auto x = torch::randn({B, L, H, D});
    auto y = attn.forward(x.view({B, L, H * D}));
    EXPECT_EQ(y.dim(), 1);
    EXPECT_EQ(y.size(0), 64);
#endif
}
