#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../trails.hpp"

using namespace trails;
using namespace trails::detail;

TEST(TensorTests, IntSequence) {
    EXPECT_EQ(int_sequence<>::length, 0);
}


