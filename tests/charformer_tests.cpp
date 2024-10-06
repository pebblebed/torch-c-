#include <gtest/gtest.h>
#include <torch/torch.h>

#include "../charformer.hpp"

using namespace trainium;

TEST(CharformerTests, PosEncoding) {
    auto enc = positional_encoding(15, 128);
    EXPECT_EQ(enc.dim(), 3);
    EXPECT_EQ(enc.size(0), 1); // Empty batch dim
    EXPECT_EQ(enc.size(1), 15);
    EXPECT_EQ(enc.size(2), 128);
}

TEST(CharformerTests, ApplyPosEncoding) {
    const auto sizes = std::tuple{11, 1024, 12, 64};
    const auto [B, L, H, D] = sizes; 
    auto x = torch::randn(sizes);
    auto enc = positional_encoding(L, D);
    EXPECT_EQ(enc.dim(), 3);
    EXPECT_EQ(enc.size(0), 1); // Empty batch dim
    EXPECT_EQ(enc.size(1), L);
    EXPECT_EQ(enc.size(2), D);
#if 0
    auto y = apply_positional_encoding(x);
    EXPECT_EQ(y.size(0), B);
    EXPECT_EQ(y.size(1), L);
    EXPECT_EQ(y.size(2), H);
    EXPECT_EQ(y.size(3), D);
#endif
}
