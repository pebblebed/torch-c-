#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../trails.hpp"

using namespace trails;
using namespace trails::detail;
using namespace trails::functional;

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

TEST(TensorTest, zeroes) {
    auto t = Tensor<>::zeroes();
    EXPECT_TRUE(t.compare_sizes(torch::IntArrayRef{}));
    EXPECT_TRUE((Tensor<1, 2, 3>::zeroes().compare_sizes(torch::IntArrayRef{1, 2, 3})));

    auto z = Tensor<1, 2, 3>::zeroes();
    for (auto i = 0; i < z.t().numel(); ++i) {
        EXPECT_EQ(z.t().data_ptr<float>()[i], 0.0f);
    }
}

TEST(TensorTest, ones) {
    auto t = Tensor<>::ones();
    EXPECT_TRUE(t.compare_sizes(torch::IntArrayRef{}));
    EXPECT_TRUE((Tensor<1, 2, 3>::ones().compare_sizes(torch::IntArrayRef{1, 2, 3})));
    EXPECT_FALSE((Tensor<2, 3>::ones().compare_sizes(torch::IntArrayRef{1, 2, 3})));
    EXPECT_FALSE((Tensor<3>::ones().compare_sizes(torch::IntArrayRef{1, 2, 3})));
    EXPECT_FALSE((Scalar::ones().compare_sizes(torch::IntArrayRef{1, 2, 3})));

    auto z = Tensor<1, 2, 3>::ones();
    for (auto i = 0; i < z.t().numel(); ++i) {
        EXPECT_EQ(z.t().data_ptr<float>()[i], 1.0f);
    }
}

TEST(TensorTest, scalar_mul) {
    typedef Tensor<1, 2, 3> T;
    auto t = T::ones();
    auto s1 = t * 2.0f;
    auto s2 = 2.0f * t;
    for (auto i = 0; i < T::numel(); ++i) {
        EXPECT_EQ(s1.t().data_ptr<float>()[i], 2.0f);
        EXPECT_EQ(s2.t().data_ptr<float>()[i], 2.0f);
    }
}

TEST(TensorTest, vector_mul) {
    auto t = Tensor<1, 2, 3>::ones() * 2.0f;
    auto t2 = Tensor<1, 2, 3>::ones() * 1.5f;
    auto t3 = t * t2;
    for (auto i = 0; i < t3.numel(); ++i) {
        EXPECT_EQ(t3.data_ptr<float>()[i], 3.0f);
    }
}

TEST(TensorTest, scalar_div) {
    auto t = Tensor<1, 2, 3>::ones();
    auto s = t / 2.0f;
    auto s2 = 2.0 / t;
    for (auto i = 0; i < s.t().numel(); ++i) {
        EXPECT_EQ(s.t().data_ptr<float>()[i], 0.5f);
        EXPECT_EQ(s2.t().data_ptr<float>()[i], 2.0f);
    }
}

TEST(TensorTest, vector_div) {
    auto t = 1.8 * Tensor<1, 2, 3>::ones();
    auto s = t / (Tensor<1, 2, 3>::ones() * 2.0f);
    for (auto i = 0; i < s.t().numel(); ++i) {
        EXPECT_EQ(s.t().data_ptr<float>()[i], 0.9f);
    }
}

TEST(TensorTest, scalar_add) {
    using T = Tensor<1, 2, 3>;
    auto t = T::arange();
    auto s = T::ones();
    auto s2 = 1.0f + t;
    for (auto i = 0; i < T::numel(); ++i) {
        EXPECT_EQ(s2.data_ptr<float>()[i], 1.0f + i);
    }
}

TEST(TensorTest, vector_add) {
    using T = Tensor<1, 2, 3>;
    auto t = T::ones() + T::arange();
    for (auto i = 0; i < T::numel(); ++i) {
        EXPECT_EQ(t.data_ptr<float>()[i], 1.0f + i);
    }
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

TEST(TensorTest, conv2d) {
    constexpr int BatchSize = 2;
    constexpr int InChannels = 3;
    constexpr int OutChannels = 5;
    constexpr int InputHeight = 10;
    constexpr int InputWidth = 17;
    constexpr int KernelHeight = 5;
    constexpr int KernelWidth = 3;
    constexpr int groups = 1;
    auto input = Tensor<BatchSize, InChannels, InputHeight, InputWidth>::randn();
    auto weights = Tensor<OutChannels, InChannels / groups, KernelHeight, KernelWidth>::ones();
    auto output = trails::conv2d(input, weights);
    std::cout << decltype(output)::seq_t() << std::endl;
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{
        BatchSize,
        OutChannels, InputHeight - (KernelHeight - 1),
        InputWidth - (KernelWidth - 1)}));
}

TEST(TensorTest, square) {
    auto t = Tensor<1, 2, 3>::arange();
    auto s = t.square();
    for (auto i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(s.data_ptr<float>()[i], i * i);
    }
}

TEST(TensorTest, abs) {
    auto t = 0.0 - Tensor<1, 2, 3>::arange();
    auto s = t.abs();
    for (auto i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(s.data_ptr<float>()[i], std::abs(i));
    }
}

template<int ...Dims>
static auto mean_test_body() {
    using T = Tensor<Dims...>;
    auto t = T::arange() + T::ones();
    auto s = t.mean();
    constexpr int n = T::numel();
    auto exp_sum = n * (n + 1) / 2;
    auto exp_mean = float(exp_sum) / n;
    EXPECT_NEAR(s.template item<float>(), exp_mean, 1e-6);
}

TEST(TensorTest, mean) {
    mean_test_body<1, 1, 1>();
    mean_test_body<1, 1, 11>();
    mean_test_body<4, 3, 17>();
}
