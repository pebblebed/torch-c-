#include "trails/trails.hpp"
#include <gtest/gtest.h>

TEST(HeaderSmokeTest, TrailsHeaderSelfContained) {
    auto out = trails::detail::str(torch::IntArrayRef{1, 2, 3});
    EXPECT_EQ(out, "[1, 2, 3]");
}
