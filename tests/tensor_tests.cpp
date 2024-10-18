#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../trails.hpp"

using namespace trails;
using namespace trails::detail;

TEST(TensorTests, ValSequence) {
    EXPECT_EQ(val_sequence<double>::length, 0);
    EXPECT_EQ((val_sequence<double, 1.0>::length), 1);
    EXPECT_EQ((val_sequence<double, 1.0, 2.0>::length), 2);
    EXPECT_EQ((val_sequence<double, 1.0, 2.0, 3.0>::length), 3);

    typedef val_sequence<int, 12, 7, -43> vals_3d;
    EXPECT_EQ(vals_3d::length, 3);
    EXPECT_EQ((vals_3d::get<0>()), 12);
    EXPECT_EQ((vals_3d::get<1>()), 7);
    EXPECT_EQ((vals_3d::get<2>()), -43);

    typedef val_sequence<int, 12, 7, -43, 1> vals_4d;
    EXPECT_EQ(vals_4d::length, 4);
    EXPECT_EQ(vals_4d::get<0>(), 12);
    EXPECT_EQ(vals_4d::get<1>(), 7);
    EXPECT_EQ(vals_4d::get<2>(), -43);
    EXPECT_EQ(vals_4d::get<3>(), 1);
}

TEST(TensorTests, ValSequenceIndexing) {
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<0>()), 12);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<1>()), 7);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<2>()), -43);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<3>()), 1);
}

TEST(TensorTests, Comparison){
    EXPECT_TRUE(Tensor<>::compare_sizes(torch::IntArrayRef{}));
    EXPECT_FALSE(Tensor<>::compare_sizes(torch::IntArrayRef{1}));
    EXPECT_FALSE(Tensor<1>::compare_sizes(torch::IntArrayRef{}));

    EXPECT_TRUE(Tensor<1>::compare_sizes(torch::IntArrayRef{1}));
    EXPECT_FALSE(Tensor<1>::compare_sizes(torch::IntArrayRef{2}));

    EXPECT_TRUE((Tensor<1, 2>::compare_sizes(torch::IntArrayRef{1, 2})));
    EXPECT_FALSE((Tensor<1, 2>::compare_sizes(torch::IntArrayRef{1, 3})));
    EXPECT_FALSE((Tensor<1, 2>::compare_sizes(torch::IntArrayRef{2, 2})));
    EXPECT_FALSE((Tensor<2, 3>::compare_sizes(torch::IntArrayRef{2, 3, 4})));
}

TEST(TensorTest, Randn) {
    auto t = Tensor<>::randn();
    EXPECT_TRUE(t.compare_sizes(torch::IntArrayRef{}));

    auto t2 = Tensor<7, 10, 128>::randn();
    EXPECT_TRUE(t2.compare_sizes(torch::IntArrayRef{7, 10, 128}));
}

TEST(TensorTest, conv1d) {
    constexpr int BatchSize = 2;
    constexpr int InChannels = 3;
    constexpr int OutChannels = 5;
    constexpr int Length = 10;
    constexpr int KernelWidth = 3;
    constexpr int groups = 1;
    auto input = Tensor<BatchSize, InChannels, Length>::randn();
    auto weights = Tensor<OutChannels, InChannels / groups, KernelWidth>::ones();
    auto output = trails::conv1d(input, weights);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{BatchSize, OutChannels, Length - 2}));
}
