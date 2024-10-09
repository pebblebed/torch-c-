#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../trails.hpp"

using namespace trails;

TEST(TensorTests, SingleDim) {
    auto d = TensorDimension<2>();
    EXPECT_EQ(d.value, 2);
    EXPECT_EQ(d.dims, 1);

    auto d2 = TensorDimension<2, 3>();
    EXPECT_EQ(d2.value, 2);
    EXPECT_EQ(d2.dims, 2);
}
