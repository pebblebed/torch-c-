#include <gtest/gtest.h>
#include <torch/torch.h>
#include <sstream>
#include "../trails.hpp"
#include "../trails_nn.hpp"

using namespace trails;
using namespace trails::detail;
namespace F = trails::functional;

TEST(TensorTests, ValSequence) {
    EXPECT_EQ(val_sequence<double>::length, 0);
    EXPECT_EQ((val_sequence<double, 1.0>::length), 1);
    EXPECT_EQ((val_sequence<double, 1.0, 2.0>::length), 2);
    EXPECT_EQ((val_sequence<double, 1.0, 2.0, 3.0>::length), 3);

    typedef val_sequence<int, 12, 7, -43> vals_3d;
    EXPECT_EQ(vals_3d::length, 3);
    EXPECT_EQ((vals_3d::get<0>::value), 12);
    EXPECT_EQ((vals_3d::get<1>::value), 7);
    EXPECT_EQ((vals_3d::get<2>::value), -43);

    typedef val_sequence<int, 12, 7, -43, 1> vals_4d;
    EXPECT_EQ(vals_4d::length, 4);
    EXPECT_EQ(vals_4d::get<0>::value, 12);
    EXPECT_EQ(vals_4d::get<1>::value, 7);
    EXPECT_EQ(vals_4d::get<2>::value, -43);
    EXPECT_EQ(vals_4d::get<3>::value, 1);
}

TEST(TensorTests, ValSequenceIndexing) {
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<0>::value), 12);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<1>::value), 7);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<2>::value), -43);
    EXPECT_EQ((val_sequence<int, 12, 7, -43, 1>::get<3>::value), 1);
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
    auto output = F::conv1d(input, weights);
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
    auto output = F::conv2d(input, weights);
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

    // ReduceDims tests.
#if 0
    auto t = Tensor<1, 2, 3, 4>::arange();
    auto s = t.mean<true>();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{1, 1, 1, 1}));
    EXPECT_EQ(s.template item<float>(), 1.5f);
#endif
}

TEST(TensorTest, reducedims) {
    using namespace trails::detail;
    {
        using T = Tensor<17, 2, 3>;
        {
            using K = ReduceDims<T, false, 1, 2>;
            K k;
            static const auto dims = K::dims;
            EXPECT_EQ(dims, (std::array<int64_t, 1>{17}));
        }
        {
            using K = ReduceDims<T, true, 1, 2>;
            K k;
            static const auto dims = K::dims;
            EXPECT_EQ(dims, (std::array<int64_t, 3>{17, 1, 1}));
        }
    }
}

TEST(TensorTest, set_dim) {
    using namespace trails::detail;
    using T0 = val_sequence<int>;
    using T01 = val_sequence<int, 1>;
    static_assert(!T0::equals<T01>::value);
    static_assert(T0::equals<T0>::value);
    static_assert(T01::equals<T01>::value);

    using T1 = val_sequence<int, 1, 2, 3>;
    using T2 = T1::set_dim<1, 4>::type;
    static_assert(T2::equals<val_sequence<int, 1, 4, 3>>::value);

    using T3 = T1::set_dim<0, 4>::type;
    static_assert(T3::equals<val_sequence<int, 4, 2, 3>>::value);
}

// ---- Wave 1A: Core Tensor operations ----

TEST(TensorTest, matmul_2d) {
    // M=2, K=3, N=4: ones(2,3) x ones(3,4) = 3*ones(2,4)
    auto a = Tensor<2, 3>::ones();
    auto b = Tensor<3, 4>::ones();
    auto c = matmul(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 4}));
    for (int i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data_ptr<float>()[i], 3.0f);
    }
}

TEST(TensorTest, matmul_2d_identity) {
    // Multiply by identity-like: arange reshaped
    auto a = Tensor<2, 2>::ones();
    auto eye = Tensor<2, 2>(torch::eye(2));
    auto c = matmul(a, eye);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 2}));
    // ones * eye = ones
    for (int i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data_ptr<float>()[i], 1.0f);
    }
}

TEST(TensorTest, matmul_batched_3d) {
    // B=2, M=3, K=4, N=5
    auto a = Tensor<2, 3, 4>::ones();
    auto b = Tensor<2, 4, 5>::ones();
    auto c = matmul(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 3, 5}));
    for (int i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data_ptr<float>()[i], 4.0f);
    }
}

TEST(TensorTest, transpose_2d) {
    auto t = Tensor<2, 3>::arange();
    auto tr = t.transpose<0, 1>();
    EXPECT_TRUE(tr.compare_sizes(torch::IntArrayRef{3, 2}));
    // Check that element [0,1] of original == element [1,0] of transposed
    auto orig_val = t.t().index({0, 1}).item<float>();
    auto trans_val = tr.t().index({1, 0}).item<float>();
    EXPECT_FLOAT_EQ(orig_val, trans_val);
}

TEST(TensorTest, transpose_3d) {
    auto t = Tensor<2, 3, 4>::randn();
    auto tr = t.transpose<0, 2>();
    EXPECT_TRUE(tr.compare_sizes(torch::IntArrayRef{4, 3, 2}));
    // Verify a specific element
    auto orig_val = t.t().index({1, 2, 3}).item<float>();
    auto trans_val = tr.t().index({3, 2, 1}).item<float>();
    EXPECT_FLOAT_EQ(orig_val, trans_val);
}

TEST(TensorTest, transpose_same_dim) {
    // Transposing same dim is identity
    auto t = Tensor<2, 3>::arange();
    auto tr = t.transpose<0, 0>();
    EXPECT_TRUE(tr.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, reshape_basic) {
    auto t = Tensor<2, 3>::arange();
    auto r = t.reshape<3, 2>();
    EXPECT_TRUE(r.compare_sizes(torch::IntArrayRef{3, 2}));
    EXPECT_EQ(r.numel(), t.numel());
}

TEST(TensorTest, reshape_flatten) {
    auto t = Tensor<2, 3, 4>::ones();
    auto r = t.reshape<24>();
    EXPECT_TRUE(r.compare_sizes(torch::IntArrayRef{24}));
}

TEST(TensorTest, view_basic) {
    auto t = Tensor<2, 3>::arange();
    auto v = t.view<6>();
    EXPECT_TRUE(v.compare_sizes(torch::IntArrayRef{6}));
    // Values should be preserved
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(v.data_ptr<float>()[i], float(i));
    }
}

TEST(TensorTest, view_expand_dims) {
    auto t = Tensor<12>::arange();
    auto v = t.view<3, 4>();
    EXPECT_TRUE(v.compare_sizes(torch::IntArrayRef{3, 4}));
}

TEST(TensorTest, unsqueeze_front) {
    auto t = Tensor<2, 3>::arange();
    auto u = t.unsqueeze<0>();
    EXPECT_TRUE(u.compare_sizes(torch::IntArrayRef{1, 2, 3}));
}

TEST(TensorTest, unsqueeze_middle) {
    auto t = Tensor<2, 3>::arange();
    auto u = t.unsqueeze<1>();
    EXPECT_TRUE(u.compare_sizes(torch::IntArrayRef{2, 1, 3}));
}

TEST(TensorTest, unsqueeze_back) {
    auto t = Tensor<2, 3>::arange();
    auto u = t.unsqueeze<2>();
    EXPECT_TRUE(u.compare_sizes(torch::IntArrayRef{2, 3, 1}));
}

TEST(TensorTest, squeeze_front) {
    auto t = Tensor<1, 2, 3>::arange();
    auto s = t.squeeze<0>();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, squeeze_middle) {
    auto t = Tensor<2, 1, 3>::arange();
    auto s = t.squeeze<1>();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, squeeze_roundtrip) {
    // unsqueeze then squeeze should give back original shape
    auto t = Tensor<2, 3>::arange();
    auto u = t.unsqueeze<1>();
    auto s = u.squeeze<1>();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));
    for (int i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(s.data_ptr<float>()[i], t.data_ptr<float>()[i]);
    }
}

TEST(TensorTest, cat_dim0) {
    auto a = Tensor<2, 3>::ones();
    auto b = Tensor<4, 3>::ones();
    auto c = F::cat<0>(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{6, 3}));
}

TEST(TensorTest, cat_dim1) {
    auto a = Tensor<2, 3>::ones();
    auto b = Tensor<2, 5>::ones();
    auto c = F::cat<1>(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 8}));
}

TEST(TensorTest, cat_3d) {
    auto a = Tensor<2, 3, 4>::arange();
    auto b = Tensor<2, 3, 6>::ones();
    auto c = F::cat<2>(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 3, 10}));
}

// ---- Wave 1B: Activation functions and elementwise ops ----

TEST(TensorTest, relu_shape_and_values) {
    // Shape preservation
    auto t = Tensor<2, 3>::randn();
    auto r = t.relu();
    EXPECT_TRUE(r.compare_sizes(torch::IntArrayRef{2, 3}));

    // Value correctness: relu(x) = max(0, x)
    auto input = Tensor<2, 3>(torch::tensor({{-1.0f, 0.0f, 1.0f}, {-2.0f, 3.0f, -0.5f}}));
    auto out = input.relu();
    auto expected = torch::tensor({{0.0f, 0.0f, 1.0f}, {0.0f, 3.0f, 0.0f}});
    EXPECT_TRUE(torch::allclose(out.t(), expected));

    // Functional version
    auto fr = F::relu(input);
    EXPECT_TRUE(torch::allclose(fr.t(), expected));
}

TEST(TensorTest, gelu_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto g = t.gelu();
    EXPECT_TRUE(g.compare_sizes(torch::IntArrayRef{2, 3}));

    // GELU(0) = 0
    auto zeros = Tensor<2, 3>::zeroes();
    auto gz = zeros.gelu();
    EXPECT_TRUE(torch::allclose(gz.t(), torch::zeros({2, 3}), 1e-5, 1e-5));

    // Functional version
    auto fg = F::gelu(t);
    EXPECT_TRUE(fg.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, sigmoid_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto s = t.sigmoid();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));

    // sigmoid(0) = 0.5
    auto zeros = Tensor<2, 3>::zeroes();
    auto sz = zeros.sigmoid();
    auto expected = torch::ones({2, 3}) * 0.5f;
    EXPECT_TRUE(torch::allclose(sz.t(), expected, 1e-5, 1e-5));

    // Functional version
    auto fs = F::sigmoid(t);
    EXPECT_TRUE(fs.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, tanh_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto th = t.tanh();
    EXPECT_TRUE(th.compare_sizes(torch::IntArrayRef{2, 3}));

    // tanh(0) = 0
    auto zeros = Tensor<2, 3>::zeroes();
    auto tz = zeros.tanh();
    EXPECT_TRUE(torch::allclose(tz.t(), torch::zeros({2, 3}), 1e-5, 1e-5));

    // Functional version
    auto ft = F::tanh(t);
    EXPECT_TRUE(ft.compare_sizes(torch::IntArrayRef{2, 3}));
}

TEST(TensorTest, softmax_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto s = t.softmax<1>();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));

    // softmax sums to 1 along dim 1
    auto sums = s.t().sum(1);
    EXPECT_TRUE(torch::allclose(sums, torch::ones({2}), 1e-5, 1e-5));

    // softmax along dim 0
    auto s0 = t.softmax<0>();
    EXPECT_TRUE(s0.compare_sizes(torch::IntArrayRef{2, 3}));
    auto sums0 = s0.t().sum(0);
    EXPECT_TRUE(torch::allclose(sums0, torch::ones({3}), 1e-5, 1e-5));

    // Functional version
    auto fs = F::softmax<1>(t);
    EXPECT_TRUE(torch::allclose(fs.t(), s.t()));
}

TEST(TensorTest, log_softmax_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto ls = t.log_softmax<1>();
    EXPECT_TRUE(ls.compare_sizes(torch::IntArrayRef{2, 3}));

    // exp(log_softmax) should sum to 1
    auto exp_ls = ls.t().exp().sum(1);
    EXPECT_TRUE(torch::allclose(exp_ls, torch::ones({2}), 1e-5, 1e-5));

    // log_softmax values should all be <= 0
    EXPECT_TRUE((ls.t() <= 0).all().item<bool>());

    // Functional version
    auto fls = F::log_softmax<1>(t);
    EXPECT_TRUE(torch::allclose(fls.t(), ls.t()));
}

TEST(TensorTest, dropout_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    // In eval mode (p=0 or training=false), dropout is identity
    auto d = F::dropout(t, /*p=*/0.5, /*training=*/false);
    EXPECT_TRUE(d.compare_sizes(torch::IntArrayRef{2, 3}));
    EXPECT_TRUE(torch::allclose(d.t(), t.t()));

    // With p=0, dropout is always identity
    auto d0 = F::dropout(t, /*p=*/0.0, /*training=*/true);
    EXPECT_TRUE(torch::allclose(d0.t(), t.t()));
}

TEST(TensorTest, exp_shape_and_values) {
    auto t = Tensor<2, 3>::randn();
    auto e = t.exp();
    EXPECT_TRUE(e.compare_sizes(torch::IntArrayRef{2, 3}));

    // exp(0) = 1
    auto zeros = Tensor<2, 3>::zeroes();
    auto ez = zeros.exp();
    EXPECT_TRUE(torch::allclose(ez.t(), torch::ones({2, 3}), 1e-5, 1e-5));

    // Functional version
    auto fe = F::exp(t);
    EXPECT_TRUE(torch::allclose(fe.t(), e.t()));
}

TEST(TensorTest, log_shape_and_values) {
    // Use positive values for log
    auto t = Tensor<2, 3>::ones();
    auto l = t.log();
    EXPECT_TRUE(l.compare_sizes(torch::IntArrayRef{2, 3}));

    // log(1) = 0
    EXPECT_TRUE(torch::allclose(l.t(), torch::zeros({2, 3}), 1e-5, 1e-5));

    // log(exp(x)) = x roundtrip
    auto x = Tensor<2, 3>::randn();
    auto roundtrip = x.exp().log();
    EXPECT_TRUE(torch::allclose(roundtrip.t(), x.t(), 1e-4, 1e-4));

    // Functional version
    auto fl = F::log(t);
    EXPECT_TRUE(torch::allclose(fl.t(), l.t()));
}

TEST(TensorTest, sqrt_shape_and_values) {
    // Use positive values for sqrt
    auto t = Tensor<2, 3>(torch::tensor({{1.0f, 4.0f, 9.0f}, {16.0f, 25.0f, 36.0f}}));
    auto s = t.sqrt();
    EXPECT_TRUE(s.compare_sizes(torch::IntArrayRef{2, 3}));

    auto expected = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    EXPECT_TRUE(torch::allclose(s.t(), expected, 1e-5, 1e-5));

    // Functional version
    auto fs = F::sqrt(t);
    EXPECT_TRUE(torch::allclose(fs.t(), expected, 1e-5, 1e-5));
}


TEST(TensorTest, LayerNorm) {
    trails::nn::LayerNorm<2, 3, 4> ln;
    auto x = Tensor<2, 3, 4>::randn();
    auto y = ln.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{2, 3, 4}));
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, 2);
    EXPECT_EQ(y.size<1>, 3);
    EXPECT_EQ(y.size<2>, 4);
}

TEST(TensorTest, BatchNorm1d) {
    trails::nn::BatchNorm1d<2, 3, 4> bn;
    bn.eval();
    auto x = Tensor<2, 3, 4>::randn();
    auto y = bn.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{2, 3, 4}));
    EXPECT_EQ(y.dim(), 3);
    EXPECT_EQ(y.size<0>, 2);
    EXPECT_EQ(y.size<1>, 3);
    EXPECT_EQ(y.size<2>, 4);
}

TEST(TensorTest, BatchNorm2d) {
    trails::nn::BatchNorm2d<2, 3, 4, 5> bn;
    bn.eval();
    auto x = Tensor<2, 3, 4, 5>::randn();
    auto y = bn.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{2, 3, 4, 5}));
    EXPECT_EQ(y.dim(), 4);
    EXPECT_EQ(y.size<0>, 2);
    EXPECT_EQ(y.size<1>, 3);
    EXPECT_EQ(y.size<2>, 4);
    EXPECT_EQ(y.size<3>, 5);
}

// ---- Wave 3B: Pooling, Embedding, and remaining functional ops ----

TEST(TensorTest, max_pool1d_basic) {
    // Input: B=2, C=3, L=10, kernel_size=3, stride=3
    // Output length: (10 - 3) / 3 + 1 = 3
    auto input = Tensor<2, 3, 10>::randn();
    auto output = F::max_pool1d<3, 3>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 3, 3}));
}

TEST(TensorTest, max_pool1d_stride1) {
    // Input: B=1, C=1, L=8, kernel_size=3, stride=1
    // Output length: (8 - 3) / 1 + 1 = 6
    auto input = Tensor<1, 1, 8>::arange();
    auto output = F::max_pool1d<3, 1>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 1, 6}));
    // First window [0,1,2] -> max=2, second [1,2,3] -> max=3, etc.
    EXPECT_FLOAT_EQ(output.t().index({0, 0, 0}).item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(output.t().index({0, 0, 1}).item<float>(), 3.0f);
}

TEST(TensorTest, max_pool2d_basic) {
    // Input: B=2, C=3, H=8, W=8, kernel=2x2, stride=2x2
    // Output: B=2, C=3, H=4, W=4
    auto input = Tensor<2, 3, 8, 8>::randn();
    auto output = F::max_pool2d<2, 2, 2, 2>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 3, 4, 4}));
}

TEST(TensorTest, max_pool2d_asymmetric) {
    // Input: B=1, C=1, H=6, W=9, kernel=3x3, stride=2x3
    // Output H: (6 - 3) / 2 + 1 = 2
    // Output W: (9 - 3) / 3 + 1 = 3
    auto input = Tensor<1, 1, 6, 9>::randn();
    auto output = F::max_pool2d<3, 3, 2, 3>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 1, 2, 3}));
}

TEST(TensorTest, avg_pool1d_basic) {
    // Input: B=2, C=3, L=10, kernel_size=2, stride=2
    // Output length: (10 - 2) / 2 + 1 = 5
    auto input = Tensor<2, 3, 10>::randn();
    auto output = F::avg_pool1d<2, 2>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 3, 5}));
}

TEST(TensorTest, avg_pool1d_values) {
    // Input: B=1, C=1, L=4, kernel_size=2, stride=2
    // [0, 1, 2, 3] -> avg([0,1])=0.5, avg([2,3])=2.5
    auto input = Tensor<1, 1, 4>::arange();
    auto output = F::avg_pool1d<2, 2>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 1, 2}));
    EXPECT_NEAR(output.t().index({0, 0, 0}).item<float>(), 0.5f, 1e-5);
    EXPECT_NEAR(output.t().index({0, 0, 1}).item<float>(), 2.5f, 1e-5);
}

TEST(TensorTest, avg_pool2d_basic) {
    // Input: B=2, C=3, H=8, W=8, kernel=2x2, stride=2x2
    // Output: B=2, C=3, H=4, W=4
    auto input = Tensor<2, 3, 8, 8>::randn();
    auto output = F::avg_pool2d<2, 2, 2, 2>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 3, 4, 4}));
}

TEST(TensorTest, avg_pool2d_asymmetric) {
    // Input: B=1, C=1, H=6, W=9, kernel=3x3, stride=2x3
    // Output H: (6 - 3) / 2 + 1 = 2
    // Output W: (9 - 3) / 3 + 1 = 3
    auto input = Tensor<1, 1, 6, 9>::randn();
    auto output = F::avg_pool2d<3, 3, 2, 3>(input);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 1, 2, 3}));
}

TEST(TensorTest, Embedding_basic) {
    // VocabSize=100, EmbedDim=32
    // Input: Tensor<2, 5> of integer indices -> Output: Tensor<2, 5, 32>
    trails::nn::Embedding<100, 32> emb;
    auto indices = Tensor<2, 5>(torch::randint(0, 100, {2, 5}, torch::kLong));
    auto output = emb.forward(indices);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 5, 32}));
}

TEST(TensorTest, Embedding_single_batch) {
    // VocabSize=10, EmbedDim=4
    trails::nn::Embedding<10, 4> emb;
    auto indices = Tensor<1, 3>(torch::randint(0, 10, {1, 3}, torch::kLong));
    auto output = emb.forward(indices);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 3, 4}));
}

TEST(TensorTest, flatten_all) {
    // Flatten all dims: Tensor<2, 3, 4> -> Tensor<24>
    auto t = Tensor<2, 3, 4>::arange();
    auto f = F::flatten<0, 2>(t);
    EXPECT_TRUE(f.compare_sizes(torch::IntArrayRef{24}));
    // Values should be preserved
    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(f.data_ptr<float>()[i], float(i));
    }
}

TEST(TensorTest, flatten_partial) {
    // Flatten dims 1,2: Tensor<2, 3, 4> -> Tensor<2, 12>
    auto t = Tensor<2, 3, 4>::randn();
    auto f = F::flatten<1, 2>(t);
    EXPECT_TRUE(f.compare_sizes(torch::IntArrayRef{2, 12}));
}

TEST(TensorTest, flatten_middle) {
    // Flatten dims 1,2 of 4D: Tensor<2, 3, 4, 5> -> Tensor<2, 12, 5>
    auto t = Tensor<2, 3, 4, 5>::randn();
    auto f = F::flatten<1, 2>(t);
    EXPECT_TRUE(f.compare_sizes(torch::IntArrayRef{2, 12, 5}));
}

TEST(TensorTest, flatten_single_dim) {
    // Flatten a single dim (no-op): Tensor<2, 3, 4> -> Tensor<2, 3, 4>
    auto t = Tensor<2, 3, 4>::randn();
    auto f = F::flatten<1, 1>(t);
    EXPECT_TRUE(f.compare_sizes(torch::IntArrayRef{2, 3, 4}));
}

TEST(TensorTest, functional_linear_no_bias) {
    // input: Tensor<2, 3>, weight: Tensor<4, 3> -> output: Tensor<2, 4>
    auto input = Tensor<2, 3>::ones();
    auto weight = Tensor<4, 3>::ones();
    auto output = F::linear(input, weight);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 4}));
    // ones(2,3) x ones(3,4) = 3*ones(2,4)
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data_ptr<float>()[i], 3.0f);
    }
}

TEST(TensorTest, functional_linear_with_bias) {
    auto input = Tensor<2, 3>::ones();
    auto weight = Tensor<4, 3>::ones();
    auto bias = Tensor<4>::ones();
    auto output = F::linear(input, weight, std::optional{bias});
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 4}));
    // ones(2,3) x ones(3,4) + ones(4) = 4*ones(2,4)
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data_ptr<float>()[i], 4.0f);
    }
}

TEST(TensorTest, functional_linear_batched) {
    // 3D input: Tensor<B, SeqLen, InDim> x Tensor<OutDim, InDim> -> Tensor<B, SeqLen, OutDim>
    auto input = Tensor<2, 5, 3>::ones();
    auto weight = Tensor<4, 3>::ones();
    auto output = F::linear(input, weight);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 5, 4}));
    for (int i = 0; i < output.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data_ptr<float>()[i], 3.0f);
    }
}

// ---- Wave 3A: Attention ----

TEST(TensorTest, matmul_batched_4d) {
    // B=2, H=3, M=4, K=5, N=6
    auto a = Tensor<2, 3, 4, 5>::ones();
    auto b = Tensor<2, 3, 5, 6>::ones();
    auto c = matmul(a, b);
    EXPECT_TRUE(c.compare_sizes(torch::IntArrayRef{2, 3, 4, 6}));
    for (int i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data_ptr<float>()[i], 5.0f);
    }
}

TEST(TensorTest, scaled_dot_product_attention_basic) {
    // B=1, H=1, L=2, D=4, S=2
    // Use identity-like Q, K so attention weights are uniform
    auto Q = Tensor<1, 1, 2, 4>::ones();
    auto K = Tensor<1, 1, 2, 4>::ones();
    auto V = Tensor<1, 1, 2, 4>::ones();
    auto out = F::scaled_dot_product_attention(Q, K, V);
    EXPECT_TRUE(out.compare_sizes(torch::IntArrayRef{1, 1, 2, 4}));
    // With uniform Q, K, V=ones, output should be ones
    for (int i = 0; i < out.numel(); ++i) {
        EXPECT_NEAR(out.data_ptr<float>()[i], 1.0f, 1e-5);
    }
}

TEST(TensorTest, scaled_dot_product_attention_shape) {
    // B=2, H=4, L=8, S=6, D=16
    auto Q = Tensor<2, 4, 8, 16>::randn();
    auto K = Tensor<2, 4, 6, 16>::randn();
    auto V = Tensor<2, 4, 6, 16>::randn();
    auto out = F::scaled_dot_product_attention(Q, K, V);
    EXPECT_TRUE(out.compare_sizes(torch::IntArrayRef{2, 4, 8, 16}));
}

TEST(TensorTest, scaled_dot_product_attention_self) {
    // Self-attention: L == S
    // B=2, H=2, L=S=4, D=8
    auto Q = Tensor<2, 2, 4, 8>::randn();
    auto K = Tensor<2, 2, 4, 8>::randn();
    auto V = Tensor<2, 2, 4, 8>::randn();
    auto out = F::scaled_dot_product_attention(Q, K, V);
    EXPECT_TRUE(out.compare_sizes(torch::IntArrayRef{2, 2, 4, 8}));
}

TEST(TensorTest, scaled_dot_product_attention_scaling) {
    // Verify scaling: with D=4, scale = 1/sqrt(4) = 0.5
    // Use specific values to verify the math
    auto Q = Tensor<1, 1, 1, 4>(torch::ones({1, 1, 1, 4}));
    auto K = Tensor<1, 1, 1, 4>(torch::ones({1, 1, 1, 4}));
    auto V = Tensor<1, 1, 1, 4>(torch::tensor({{{{1.0f, 2.0f, 3.0f, 4.0f}}}}));
    auto out = F::scaled_dot_product_attention(Q, K, V);
    // Single key, so attention weight is 1.0 on that key
    // Output should equal V
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out.data_ptr<float>()[i], V.data_ptr<float>()[i], 1e-5);
    }
}

TEST(TensorTest, MultiHeadAttention_shape) {
    // B=2, SeqLen=8, NumHeads=2, ModelDim=16
    // HeadDim = ModelDim / NumHeads = 8
    trails::nn::MultiHeadAttention<2, 8, 2, 16> mha;
    auto x = Tensor<2, 8, 16>::randn();
    auto y = mha.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{2, 8, 16}));
}

TEST(TensorTest, MultiHeadAttention_single_head) {
    // B=1, SeqLen=4, NumHeads=1, ModelDim=8
    trails::nn::MultiHeadAttention<1, 4, 1, 8> mha;
    auto x = Tensor<1, 4, 8>::randn();
    auto y = mha.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{1, 4, 8}));
}

TEST(TensorTest, MultiHeadAttention_multi_head) {
    // B=2, SeqLen=6, NumHeads=4, ModelDim=32
    trails::nn::MultiHeadAttention<2, 6, 4, 32> mha;
    auto x = Tensor<2, 6, 32>::randn();
    auto y = mha.forward(x);
    EXPECT_TRUE(y.compare_sizes(torch::IntArrayRef{2, 6, 32}));
}

// ============================================================
// Recurrent layers: RNN, LSTM, GRU
// ============================================================

TEST(TensorTest, RNN_basic) {
    // B=2, SeqLen=5, InputSize=10, HiddenSize=20, NumLayers=1
    trails::nn::RNN<2, 5, 10, 20, 1> rnn;
    auto x = Tensor<2, 5, 10>::randn();
    auto [output, h_n] = rnn.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 5, 20}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{1, 2, 20}));
}

TEST(TensorTest, RNN_multi_layer) {
    // B=3, SeqLen=7, InputSize=8, HiddenSize=16, NumLayers=3
    trails::nn::RNN<3, 7, 8, 16, 3> rnn;
    auto x = Tensor<3, 7, 8>::randn();
    auto [output, h_n] = rnn.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{3, 7, 16}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{3, 3, 16}));
}

TEST(TensorTest, LSTM_basic) {
    // B=2, SeqLen=5, InputSize=10, HiddenSize=20, NumLayers=1
    trails::nn::LSTM<2, 5, 10, 20, 1> lstm;
    auto x = Tensor<2, 5, 10>::randn();
    auto [output, h_n, c_n] = lstm.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 5, 20}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{1, 2, 20}));
    EXPECT_TRUE(c_n.compare_sizes(torch::IntArrayRef{1, 2, 20}));
}

TEST(TensorTest, LSTM_multi_layer) {
    // B=4, SeqLen=6, InputSize=12, HiddenSize=24, NumLayers=2
    trails::nn::LSTM<4, 6, 12, 24, 2> lstm;
    auto x = Tensor<4, 6, 12>::randn();
    auto [output, h_n, c_n] = lstm.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{4, 6, 24}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{2, 4, 24}));
    EXPECT_TRUE(c_n.compare_sizes(torch::IntArrayRef{2, 4, 24}));
}

TEST(TensorTest, GRU_basic) {
    // B=2, SeqLen=5, InputSize=10, HiddenSize=20, NumLayers=1
    trails::nn::GRU<2, 5, 10, 20, 1> gru;
    auto x = Tensor<2, 5, 10>::randn();
    auto [output, h_n] = gru.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{2, 5, 20}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{1, 2, 20}));
}

TEST(TensorTest, GRU_multi_layer) {
    // B=3, SeqLen=7, InputSize=8, HiddenSize=16, NumLayers=3
    trails::nn::GRU<3, 7, 8, 16, 3> gru;
    auto x = Tensor<3, 7, 8>::randn();
    auto [output, h_n] = gru.forward(x);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{3, 7, 16}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{3, 3, 16}));
}

// ============================================================
// Wave 4: Integration tests — end-to-end small models
// ============================================================

TEST(IntegrationTest, ConvNet_MNIST) {
    // Simple ConvNet (MNIST-style):
    // Conv2d(1->16, 5x5) -> relu -> MaxPool2d(2x2) -> Conv2d(16->32, 5x5) -> relu -> Flatten -> Linear
    // Input: Tensor<1, 1, 28, 28> (batch=1, channels=1, 28x28)

    // Conv2d layer 1: 1 input channel, 16 output channels, 5x5 kernel
    // Output: (28 - 5 + 1) = 24 -> Tensor<1, 16, 24, 24>
    auto conv1_weight = Tensor<16, 1, 5, 5>::randn();

    // MaxPool2d: 2x2 kernel, stride 2
    // Output: (24 - 2) / 2 + 1 = 12 -> Tensor<1, 16, 12, 12>

    // Conv2d layer 2: 16 input channels, 32 output channels, 5x5 kernel
    // Output: (12 - 5 + 1) = 8 -> Tensor<1, 32, 8, 8>
    auto conv2_weight = Tensor<32, 16, 5, 5>::randn();

    // Flatten dims 1-3: Tensor<1, 32*8*8> = Tensor<1, 2048>
    // Linear: 2048 -> 10 (10 classes)
    trails::nn::Linear<1, 2048, 10> fc;

    // Forward pass
    auto input = Tensor<1, 1, 28, 28>::randn();

    // Conv1 + relu
    auto x = F::conv2d(input, conv1_weight);
    EXPECT_TRUE(x.compare_sizes(torch::IntArrayRef{1, 16, 24, 24}));
    auto x_relu1 = F::relu(x);

    // MaxPool2d
    auto x_pool = F::max_pool2d<2, 2, 2, 2>(x_relu1);
    EXPECT_TRUE(x_pool.compare_sizes(torch::IntArrayRef{1, 16, 12, 12}));

    // Conv2 + relu
    auto x2 = F::conv2d(x_pool, conv2_weight);
    EXPECT_TRUE(x2.compare_sizes(torch::IntArrayRef{1, 32, 8, 8}));
    auto x_relu2 = F::relu(x2);

    // Flatten
    auto x_flat = F::flatten<1, 3>(x_relu2);
    EXPECT_TRUE(x_flat.compare_sizes(torch::IntArrayRef{1, 2048}));

    // Linear classifier
    auto output = fc.forward(x_flat);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{1, 10}));

    // Verify softmax produces valid probabilities
    auto probs = output.softmax<1>();
    auto sum = probs.t().sum(1);
    EXPECT_TRUE(torch::allclose(sum, torch::ones({1}), 1e-5, 1e-5));
}

TEST(IntegrationTest, RNN_TextClassification) {
    // Simple RNN (text classification):
    // Embedding -> LSTM -> take last hidden -> Linear
    // Input: Tensor<2, 10> (batch=2, seq_len=10, long indices)

    constexpr int B = 2;
    constexpr int SeqLen = 10;
    constexpr int VocabSize = 50;
    constexpr int EmbedDim = 32;
    constexpr int HiddenSize = 64;
    constexpr int NumClasses = 5;

    trails::nn::Embedding<VocabSize, EmbedDim> emb;
    trails::nn::LSTM<B, SeqLen, EmbedDim, HiddenSize, 1> lstm;
    trails::nn::Linear<B, HiddenSize, NumClasses> classifier;

    // Input: random integer indices in [0, VocabSize)
    auto indices = Tensor<B, SeqLen>(torch::randint(0, VocabSize, {B, SeqLen}, torch::kLong));

    // Embedding: (B, SeqLen) -> (B, SeqLen, EmbedDim)
    auto embedded = emb.forward(indices);
    EXPECT_TRUE(embedded.compare_sizes(torch::IntArrayRef{B, SeqLen, EmbedDim}));

    // LSTM: (B, SeqLen, EmbedDim) -> output (B, SeqLen, HiddenSize), h_n (1, B, HiddenSize)
    auto [lstm_out, h_n, c_n] = lstm.forward(embedded);
    EXPECT_TRUE(lstm_out.compare_sizes(torch::IntArrayRef{B, SeqLen, HiddenSize}));
    EXPECT_TRUE(h_n.compare_sizes(torch::IntArrayRef{1, B, HiddenSize}));

    // Take last hidden state: h_n is (1, B, HiddenSize), squeeze to (B, HiddenSize)
    auto last_hidden = Tensor<B, HiddenSize>(h_n.t().squeeze(0));
    EXPECT_TRUE(last_hidden.compare_sizes(torch::IntArrayRef{B, HiddenSize}));

    // Linear classifier: (B, HiddenSize) -> (B, NumClasses)
    auto logits = classifier.forward(last_hidden);
    EXPECT_TRUE(logits.compare_sizes(torch::IntArrayRef{B, NumClasses}));

    // Verify softmax produces valid probabilities per batch element
    auto probs = logits.softmax<1>();
    auto sums = probs.t().sum(1);
    EXPECT_TRUE(torch::allclose(sums, torch::ones({B}), 1e-5, 1e-5));
}

TEST(IntegrationTest, TransformerEncoderBlock) {
    // Transformer encoder block:
    // MultiHeadAttention -> residual + LayerNorm -> FFN (Linear+relu+Linear) -> residual + LayerNorm
    // Input: Tensor<2, 8, 64> (batch=2, seq_len=8, model_dim=64)

    constexpr int B = 2;
    constexpr int SeqLen = 8;
    constexpr int ModelDim = 64;
    constexpr int NumHeads = 4;
    constexpr int FFDim = 256;

    trails::nn::MultiHeadAttention<B, SeqLen, NumHeads, ModelDim> mha;
    trails::nn::LayerNorm<B, SeqLen, ModelDim> ln1;
    trails::nn::LayerNorm<B, SeqLen, ModelDim> ln2;

    // FFN weights
    auto ff_w1 = Tensor<FFDim, ModelDim>::randn();
    auto ff_b1 = Tensor<FFDim>::randn();
    auto ff_w2 = Tensor<ModelDim, FFDim>::randn();
    auto ff_b2 = Tensor<ModelDim>::randn();

    auto input = Tensor<B, SeqLen, ModelDim>::randn();

    // Self-attention sublayer
    auto attn_out = mha.forward(input);
    EXPECT_TRUE(attn_out.compare_sizes(torch::IntArrayRef{B, SeqLen, ModelDim}));

    // Residual connection + LayerNorm
    auto x1 = ln1.forward(attn_out + input);
    EXPECT_TRUE(x1.compare_sizes(torch::IntArrayRef{B, SeqLen, ModelDim}));

    // Feedforward sublayer: Linear(64->256) -> relu -> Linear(256->64)
    auto ff_hidden = F::linear(x1, ff_w1, std::optional{ff_b1});
    EXPECT_TRUE(ff_hidden.compare_sizes(torch::IntArrayRef{B, SeqLen, FFDim}));

    auto ff_relu = F::relu(ff_hidden);
    EXPECT_TRUE(ff_relu.compare_sizes(torch::IntArrayRef{B, SeqLen, FFDim}));

    auto ff_out = F::linear(ff_relu, ff_w2, std::optional{ff_b2});
    EXPECT_TRUE(ff_out.compare_sizes(torch::IntArrayRef{B, SeqLen, ModelDim}));

    // Residual connection + LayerNorm
    auto output = ln2.forward(ff_out + x1);
    EXPECT_TRUE(output.compare_sizes(torch::IntArrayRef{B, SeqLen, ModelDim}));

    // Verify output is finite (no NaN/Inf from the attention computation)
    EXPECT_TRUE(torch::isfinite(output.t()).all().item<bool>());
}

// ============================================================
// BatchTensor tests
// ============================================================

TEST(BatchTensorTest, Construction) {
    auto raw = torch::randn({7, 3, 4});
    BatchTensor<3, 4> bt(raw);
    ASSERT_EQ(bt.batch_size(), 7);
    ASSERT_EQ(bt.t().size(0), 7);
    ASSERT_EQ(bt.t().size(1), 3);
    ASSERT_EQ(bt.t().size(2), 4);
}

TEST(BatchTensorTest, ConstructionValidation) {
    auto wrong = torch::randn({7, 3, 5});  // 5 != 4
    EXPECT_THROW((BatchTensor<3, 4>(wrong)), std::runtime_error);
}

TEST(BatchTensorTest, Relu) {
    auto raw = torch::randn({5, 8});
    BatchTensor<8> bt(raw);
    auto out = bt.relu();
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
    // All values should be >= 0
    EXPECT_TRUE((out.t() >= 0).all().item<bool>());
}

TEST(BatchTensorTest, Flatten) {
    // BatchTensor<3, 4> flatten<0,1> → BatchTensor<12>
    auto raw = torch::randn({7, 3, 4});
    BatchTensor<3, 4> bt(raw);
    auto flat = F::flatten<0, 1>(bt);
    ASSERT_EQ(flat.batch_size(), 7);
    ASSERT_EQ(flat.t().size(0), 7);
    ASSERT_EQ(flat.t().size(1), 12);
}

TEST(BatchTensorTest, Conv2d) {
    // BatchTensor<1, 8, 8> through conv2d with 3x3 kernel, 4 out channels
    auto raw = torch::randn({5, 1, 8, 8});
    BatchTensor<1, 8, 8> bt(raw);
    auto weight = Tensor<4, 1, 3, 3>(torch::randn({4, 1, 3, 3}));
    auto out = F::conv2d<1, 4, 8, 8, 3, 3>(bt, weight);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 4);  // out channels
    ASSERT_EQ(out.t().size(2), 6);  // 8 - 3 + 1 = 6
    ASSERT_EQ(out.t().size(3), 6);
}

TEST(BatchTensorTest, MatmulWeightSharing) {
    // BatchTensor<4, 8> x Tensor<8, 16> → BatchTensor<4, 16>
    auto a = BatchTensor<4, 8>(torch::randn({7, 4, 8}));
    auto b = Tensor<8, 16>(torch::randn({8, 16}));
    auto out = matmul(a, b);
    ASSERT_EQ(out.batch_size(), 7);
    ASSERT_EQ(out.t().size(1), 4);
    ASSERT_EQ(out.t().size(2), 16);
}

TEST(BatchTensorTest, MatmulPerSample) {
    // BatchTensor<4, 8> x BatchTensor<8, 16> → BatchTensor<4, 16>
    auto a = BatchTensor<4, 8>(torch::randn({7, 4, 8}));
    auto b = BatchTensor<8, 16>(torch::randn({7, 8, 16}));
    auto out = matmul(a, b);
    ASSERT_EQ(out.batch_size(), 7);
    ASSERT_EQ(out.t().size(1), 4);
    ASSERT_EQ(out.t().size(2), 16);
}

TEST(BatchTensorTest, BroadcastAdd) {
    // BatchTensor<8> + Tensor<8> → BatchTensor<8>
    auto bt = BatchTensor<8>(torch::ones({5, 8}));
    auto t  = Tensor<8>(torch::ones({8}));
    auto out = bt + t;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 80.0f, 1e-4f);  // 5*8*2 = 80
}

TEST(BatchTensorTest, Bind) {
    auto raw = torch::randn({4, 6});
    BatchTensor<6> bt(raw);
    auto bound = bt.bind<4>();
    ASSERT_EQ(bound.t().size(0), 4);
    ASSERT_EQ(bound.t().size(1), 6);
    // Wrong batch size should throw
    EXPECT_THROW(bt.bind<5>(), std::runtime_error);
}

// ============================================================
// Task 1: BatchTensor activations, elementwise math, shape ops
// ============================================================

TEST(BatchTensorTest, Gelu) {
    auto raw = torch::randn({5, 8});
    BatchTensor<8> bt(raw);
    auto out = bt.gelu();
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
    // GELU(0) ≈ 0
    auto zeros = BatchTensor<4>(torch::zeros({3, 4}));
    auto gz = zeros.gelu();
    EXPECT_NEAR(gz.t().sum().item<float>(), 0.0f, 1e-5f);
}

TEST(BatchTensorTest, Sigmoid) {
    auto raw = torch::randn({5, 8});
    BatchTensor<8> bt(raw);
    auto out = bt.sigmoid();
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
    // Sigmoid output ∈ (0, 1)
    EXPECT_TRUE((out.t() > 0).all().item<bool>());
    EXPECT_TRUE((out.t() < 1).all().item<bool>());
}

TEST(BatchTensorTest, Tanh) {
    auto raw = torch::randn({5, 8});
    BatchTensor<8> bt(raw);
    auto out = bt.tanh();
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
    // Tanh output ∈ (-1, 1)
    EXPECT_TRUE((out.t() > -1).all().item<bool>());
    EXPECT_TRUE((out.t() < 1).all().item<bool>());
}

TEST(BatchTensorTest, Softmax) {
    auto raw = torch::randn({5, 10});
    BatchTensor<10> bt(raw);
    auto out = bt.softmax<0>();  // softmax over math dim 0
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 10);
    // Each row should sum to 1
    auto sums = out.t().sum(1);  // sum over dim 1 (the math dim)
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(sums[i].item<float>(), 1.0f, 1e-5f);
    }
}

TEST(BatchTensorTest, LogSoftmax) {
    auto raw = torch::randn({5, 10});
    BatchTensor<10> bt(raw);
    auto out = bt.log_softmax<0>();
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 10);
    // exp(log_softmax) should sum to 1
    auto sums = out.t().exp().sum(1);
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(sums[i].item<float>(), 1.0f, 1e-5f);
    }
}

TEST(BatchTensorTest, Exp) {
    auto raw = torch::zeros({3, 4});
    BatchTensor<4> bt(raw);
    auto out = bt.exp();
    ASSERT_EQ(out.batch_size(), 3);
    // exp(0) = 1
    EXPECT_NEAR(out.t().sum().item<float>(), 12.0f, 1e-5f);  // 3*4 = 12
}

TEST(BatchTensorTest, Log) {
    auto raw = torch::ones({3, 4});
    BatchTensor<4> bt(raw);
    auto out = bt.log();
    ASSERT_EQ(out.batch_size(), 3);
    // log(1) = 0
    EXPECT_NEAR(out.t().sum().item<float>(), 0.0f, 1e-5f);
}

TEST(BatchTensorTest, Sqrt) {
    auto raw = torch::ones({3, 4}) * 4.0f;
    BatchTensor<4> bt(raw);
    auto out = bt.sqrt();
    ASSERT_EQ(out.batch_size(), 3);
    // sqrt(4) = 2
    EXPECT_NEAR(out.t().sum().item<float>(), 24.0f, 1e-5f);  // 3*4*2 = 24
}

TEST(BatchTensorTest, BatchMean) {
    auto raw = torch::ones({5, 8}) * 3.0f;
    BatchTensor<8> bt(raw);
    auto mean = bt.batch_mean();
    // Result is Tensor<8> (no batch dim)
    ASSERT_EQ(mean.dim(), 1);
    ASSERT_EQ(mean.t().size(0), 8);
    // mean of all-3s is 3
    EXPECT_NEAR(mean.t().sum().item<float>(), 24.0f, 1e-5f);  // 8*3
}

TEST(BatchTensorTest, Reshape) {
    auto raw = torch::randn({5, 3, 4});
    BatchTensor<3, 4> bt(raw);
    auto reshaped = bt.reshape<12>();
    ASSERT_EQ(reshaped.batch_size(), 5);
    ASSERT_EQ(reshaped.t().size(0), 5);
    ASSERT_EQ(reshaped.t().size(1), 12);
    // Also test reshape to higher dims
    auto reshaped2 = bt.reshape<2, 6>();
    ASSERT_EQ(reshaped2.t().size(1), 2);
    ASSERT_EQ(reshaped2.t().size(2), 6);
}

TEST(BatchTensorTest, Transpose) {
    auto raw = torch::randn({5, 3, 4});
    BatchTensor<3, 4> bt(raw);
    auto transposed = bt.transpose<0, 1>();
    ASSERT_EQ(transposed.batch_size(), 5);
    ASSERT_EQ(transposed.t().size(0), 5);
    ASSERT_EQ(transposed.t().size(1), 4);  // swapped
    ASSERT_EQ(transposed.t().size(2), 3);  // swapped
}

TEST(BatchTensorTest, StreamOutput) {
    auto raw = torch::ones({2, 3});
    BatchTensor<3> bt(raw);
    std::ostringstream oss;
    oss << bt;
    std::string str = oss.str();
    // Should contain "BatchTensor[B=2]"
    EXPECT_NE(str.find("BatchTensor[B=2]"), std::string::npos);
}

// ============================================================
// Task 2: BatchTensor arithmetic ops
// ============================================================

TEST(BatchTensorTest, AddBatchBatch) {
    auto a = BatchTensor<4>(torch::ones({3, 4}));
    auto b = BatchTensor<4>(torch::ones({3, 4}) * 2.0f);
    auto out = a + b;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 36.0f, 1e-5f);  // 3*4*3
}

TEST(BatchTensorTest, SubBatchBatch) {
    auto a = BatchTensor<4>(torch::ones({3, 4}) * 5.0f);
    auto b = BatchTensor<4>(torch::ones({3, 4}) * 2.0f);
    auto out = a - b;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 36.0f, 1e-5f);  // 3*4*3
}

TEST(BatchTensorTest, MulBatchBatch) {
    auto a = BatchTensor<4>(torch::ones({3, 4}) * 3.0f);
    auto b = BatchTensor<4>(torch::ones({3, 4}) * 2.0f);
    auto out = a * b;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 72.0f, 1e-5f);  // 3*4*6
}

TEST(BatchTensorTest, DivBatchBatch) {
    auto a = BatchTensor<4>(torch::ones({3, 4}) * 6.0f);
    auto b = BatchTensor<4>(torch::ones({3, 4}) * 2.0f);
    auto out = a / b;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 36.0f, 1e-5f);  // 3*4*3
}

TEST(BatchTensorTest, ScalarMul) {
    auto a = BatchTensor<4>(torch::ones({3, 4}));
    auto out = a * 5.0f;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 60.0f, 1e-5f);  // 3*4*5
}

TEST(BatchTensorTest, ScalarDiv) {
    auto a = BatchTensor<4>(torch::ones({3, 4}) * 10.0f);
    auto out = a / 2.0f;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 60.0f, 1e-5f);  // 3*4*5
}

TEST(BatchTensorTest, ScalarMulReverse) {
    auto a = BatchTensor<4>(torch::ones({3, 4}));
    auto out = 5.0f * a;
    ASSERT_EQ(out.batch_size(), 3);
    EXPECT_NEAR(out.t().sum().item<float>(), 60.0f, 1e-5f);
}

TEST(BatchTensorTest, BroadcastSub) {
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto t  = Tensor<8>(torch::ones({8}));
    auto out = bt - t;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 80.0f, 1e-5f);  // 5*8*2
}

TEST(BatchTensorTest, BroadcastMul) {
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto t  = Tensor<8>(torch::ones({8}) * 2.0f);
    auto out = bt * t;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 240.0f, 1e-5f);  // 5*8*6
}

TEST(BatchTensorTest, BroadcastDiv) {
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 6.0f);
    auto t  = Tensor<8>(torch::ones({8}) * 2.0f);
    auto out = bt / t;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 120.0f, 1e-5f);  // 5*8*3
}

TEST(BatchTensorTest, BroadcastReverseAdd) {
    auto t  = Tensor<8>(torch::ones({8}) * 2.0f);
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto out = t + bt;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 200.0f, 1e-5f);  // 5*8*5
}

TEST(BatchTensorTest, BroadcastReverseSub) {
    auto t  = Tensor<8>(torch::ones({8}) * 10.0f);
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto out = t - bt;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 280.0f, 1e-5f);  // 5*8*7
}

TEST(BatchTensorTest, BroadcastReverseMul) {
    auto t  = Tensor<8>(torch::ones({8}) * 4.0f);
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto out = t * bt;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 480.0f, 1e-5f);  // 5*8*12
}

TEST(BatchTensorTest, BroadcastReverseDiv) {
    auto t  = Tensor<8>(torch::ones({8}) * 12.0f);
    auto bt = BatchTensor<8>(torch::ones({5, 8}) * 3.0f);
    auto out = t / bt;
    ASSERT_EQ(out.batch_size(), 5);
    EXPECT_NEAR(out.t().sum().item<float>(), 160.0f, 1e-5f);  // 5*8*4
}

// ============================================================
// Task 3: BatchTensor conversion (unbatch)
// ============================================================

TEST(BatchTensorTest, UnbatchMember) {
    // Tensor<4, 8>::unbatch() → BatchTensor<8> with batch_size=4
    auto t = Tensor<4, 8>(torch::randn({4, 8}));
    auto bt = t.unbatch();
    ASSERT_EQ(bt.batch_size(), 4);
    ASSERT_EQ(bt.t().size(1), 8);
    // Data should be identical
    EXPECT_TRUE(torch::equal(bt.t(), t.t()));
}

TEST(BatchTensorTest, UnbatchFreeFunction) {
    // unbatch(Tensor<4, 8>) → BatchTensor<8>
    auto t = Tensor<4, 8>(torch::randn({4, 8}));
    auto bt = unbatch(t);
    ASSERT_EQ(bt.batch_size(), 4);
    ASSERT_EQ(bt.t().size(1), 8);
    EXPECT_TRUE(torch::equal(bt.t(), t.t()));
}

TEST(BatchTensorTest, UnbatchBindRoundtrip) {
    // Tensor<4, 8> → unbatch → BatchTensor<8> → bind<4> → Tensor<4, 8>
    auto t = Tensor<4, 8>(torch::randn({4, 8}));
    auto bt = t.unbatch();
    auto t2 = bt.bind<4>();
    EXPECT_TRUE(torch::equal(t.t(), t2.t()));
}

TEST(BatchTensorTest, Unbatch3D) {
    // Tensor<3, 4, 5>::unbatch() → BatchTensor<4, 5> with batch_size=3
    auto t = Tensor<3, 4, 5>(torch::randn({3, 4, 5}));
    auto bt = t.unbatch();
    ASSERT_EQ(bt.batch_size(), 3);
    ASSERT_EQ(bt.t().size(1), 4);
    ASSERT_EQ(bt.t().size(2), 5);
    EXPECT_TRUE(torch::equal(bt.t(), t.t()));
}

// ============================================================
// Task 4: BatchTensor functional ops and cross_entropy
// ============================================================

TEST(BatchTensorTest, FunctionalLinear1D) {
    // F::linear(BatchTensor<16>, Tensor<8, 16>) → BatchTensor<8>
    auto input = BatchTensor<16>(torch::randn({5, 16}));
    auto weight = Tensor<8, 16>(torch::randn({8, 16}));
    auto out = F::linear(input, weight);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
}

TEST(BatchTensorTest, FunctionalLinear1DWithBias) {
    auto input = BatchTensor<16>(torch::randn({5, 16}));
    auto weight = Tensor<8, 16>(torch::randn({8, 16}));
    auto bias = std::optional<Tensor<8>>(Tensor<8>(torch::randn({8})));
    auto out = F::linear(input, weight, bias);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
}

TEST(BatchTensorTest, FunctionalLinear2D) {
    // F::linear(BatchTensor<4, 16>, Tensor<8, 16>) → BatchTensor<4, 8>
    auto input = BatchTensor<4, 16>(torch::randn({5, 4, 16}));
    auto weight = Tensor<8, 16>(torch::randn({8, 16}));
    auto out = F::linear(input, weight);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 4);
    ASSERT_EQ(out.t().size(2), 8);
}

TEST(BatchTensorTest, FunctionalMaxPool2d) {
    // BatchTensor<1, 8, 8> → max_pool2d 2x2 stride 2 → BatchTensor<1, 4, 4>
    auto input = BatchTensor<1, 8, 8>(torch::randn({5, 1, 8, 8}));
    auto out = F::max_pool2d<2, 2, 2, 2>(input);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 1);
    ASSERT_EQ(out.t().size(2), 4);
    ASSERT_EQ(out.t().size(3), 4);
}

TEST(BatchTensorTest, FunctionalAvgPool2d) {
    // BatchTensor<1, 8, 8> → avg_pool2d 2x2 stride 2 → BatchTensor<1, 4, 4>
    auto input = BatchTensor<1, 8, 8>(torch::ones({5, 1, 8, 8}) * 2.0f);
    auto out = F::avg_pool2d<2, 2, 2, 2>(input);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(2), 4);
    ASSERT_EQ(out.t().size(3), 4);
    // avg of all-2s is 2
    EXPECT_NEAR(out.t().mean().item<float>(), 2.0f, 1e-5f);
}

TEST(BatchTensorTest, CrossEntropyDynamicBatch) {
    // BatchTensor<10> logits + torch::Tensor labels → scalar
    auto logits = BatchTensor<10>(torch::randn({5, 10}));
    auto labels = torch::randint(0, 10, {5}, torch::kLong);
    auto loss = F::cross_entropy(logits, labels);
    ASSERT_EQ(loss.dim(), 0);  // scalar
    EXPECT_GT(loss.item<float>(), 0.0f);
}

TEST(BatchTensorTest, CrossEntropyStaticBatch) {
    // Tensor<4, 10> logits + Tensor<4> labels → scalar
    auto logits = Tensor<4, 10>(torch::randn({4, 10}));
    auto labels = Tensor<4>(torch::randint(0, 10, {4}, torch::kLong));
    auto loss = F::cross_entropy(logits, labels);
    ASSERT_EQ(loss.dim(), 0);
    EXPECT_GT(loss.item<float>(), 0.0f);
}

// ============================================================
// Task 5: Batch-agnostic nn module tests
// ============================================================

TEST(BatchAgnosticTest, LinearBasic) {
    trails::nn::Linear<16, 8> linear;
    auto input = BatchTensor<16>(torch::randn({5, 16}));
    auto out = linear.forward(input);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 8);
}

TEST(BatchAgnosticTest, LinearDifferentBatchSizes) {
    trails::nn::Linear<16, 8> linear;
    // Same module, different batch sizes
    auto out3 = linear.forward(BatchTensor<16>(torch::randn({3, 16})));
    ASSERT_EQ(out3.batch_size(), 3);
    auto out7 = linear.forward(BatchTensor<16>(torch::randn({7, 16})));
    ASSERT_EQ(out7.batch_size(), 7);
    auto out1 = linear.forward(BatchTensor<16>(torch::randn({1, 16})));
    ASSERT_EQ(out1.batch_size(), 1);
}

TEST(BatchAgnosticTest, LinearParameters) {
    trails::nn::Linear<16, 8> linear;
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2u);  // weight + bias
    // Weight shape: [8, 16], bias shape: [8]
    EXPECT_EQ(params[0].size(0), 8);
    EXPECT_EQ(params[0].size(1), 16);
    EXPECT_EQ(params[1].size(0), 8);
}

TEST(BatchAgnosticTest, LayerNormBasic) {
    trails::nn::LayerNorm<32> ln;
    auto input = BatchTensor<32>(torch::randn({5, 32}));
    auto out = ln.forward(input);
    ASSERT_EQ(out.batch_size(), 5);
    ASSERT_EQ(out.t().size(1), 32);
}

TEST(BatchAgnosticTest, LayerNormNormalization) {
    trails::nn::LayerNorm<64> ln;
    auto input = BatchTensor<64>(torch::randn({4, 64}) * 10.0f + 5.0f);
    auto out = ln.forward(input);
    // After LayerNorm, per-sample mean ≈ 0, var ≈ 1
    for (int i = 0; i < 4; i++) {
        auto sample = out.t()[i];
        EXPECT_NEAR(sample.mean().item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(sample.var().item<float>(), 1.0f, 0.02f);
    }
}


// ============================================================
// Edge cases: Tensor construction, methods, and misc
// ============================================================

TEST(EdgeCaseTest, TensorDefaultConstructor) {
    // Default constructor fills with randn — just check shape is right
    Tensor<3, 4> t;
    ASSERT_EQ(t.t().size(0), 3);
    ASSERT_EQ(t.t().size(1), 4);
    ASSERT_EQ(t.t().dim(), 2);
}

TEST(EdgeCaseTest, TensorConstructorMismatch) {
    // Wrong shape should throw
    auto raw = torch::randn({3, 5});
    EXPECT_THROW((Tensor<3, 4>(raw)), std::runtime_error);
    // Wrong number of dims
    auto raw2 = torch::randn({3, 4, 5});
    EXPECT_THROW((Tensor<3, 4>(raw2)), std::runtime_error);
    // Correct shape should not throw
    auto raw3 = torch::randn({3, 4});
    EXPECT_NO_THROW((Tensor<3, 4>(raw3)));
}

TEST(EdgeCaseTest, ArangeDefault) {
    // arange(start=0) → [0, 1, 2, ..., numel-1] reshaped
    auto t0 = Tensor<2, 3>::arange(0);
    ASSERT_EQ(t0.t().size(0), 2);
    ASSERT_EQ(t0.t().size(1), 3);
    EXPECT_NEAR(t0.t()[0][0].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(t0.t()[0][1].item<float>(), 1.0f, 1e-5f);
    EXPECT_NEAR(t0.t()[1][2].item<float>(), 5.0f, 1e-5f);

    // Default arg (start=0) should be equivalent
    auto t_default = Tensor<2, 3>::arange();
    EXPECT_NEAR(t_default.t()[0][0].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(t_default.t()[1][2].item<float>(), 5.0f, 1e-5f);

    // 1D arange
    auto t1d = Tensor<5>::arange();
    ASSERT_EQ(t1d.t().size(0), 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(t1d.t()[i].item<float>(), (float)i, 1e-5f);
    }

    // Note: arange(start>0) doesn't work for small tensors since it generates
    // fewer than numel() elements — this is a known API limitation.
}

TEST(EdgeCaseTest, Rsqrt) {
    auto t = Tensor<4>(torch::ones({4}) * 4.0f);
    auto out = t.rsqrt();
    // rsqrt(4) = 1/sqrt(4) = 0.5
    ASSERT_EQ(out.t().size(0), 4);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(out.t()[i].item<float>(), 0.5f, 1e-5f);
    }
}

TEST(EdgeCaseTest, Max) {
    auto t = Tensor<3, 4>(torch::arange(12.0f).reshape({3, 4}));
    auto m = t.max();
    // max() returns Tensor<> (scalar)
    ASSERT_EQ(m.dim(), 0);
    EXPECT_NEAR(m.t().item<float>(), 11.0f, 1e-5f);
}

TEST(EdgeCaseTest, TensorStr) {
    auto t = Tensor<2>(torch::ones({2}));
    std::string s = t.str();
    // Should contain the tensor values — "1" should appear
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("1"), std::string::npos);
}

TEST(EdgeCaseTest, TensorStreamOutput) {
    auto t = Tensor<2>(torch::ones({2}));
    std::ostringstream oss;
    oss << t;
    EXPECT_FALSE(oss.str().empty());
}

TEST(EdgeCaseTest, Item) {
    // Single-element tensor → item<float>()
    auto t = Tensor<1>(torch::ones({1}) * 42.0f);
    EXPECT_NEAR(t.item<float>(), 42.0f, 1e-5f);
    EXPECT_EQ(t.item<int>(), 42);
}

TEST(EdgeCaseTest, DataPtr) {
    auto t = Tensor<4>(torch::ones({4}) * 7.0f);
    const float* ptr = t.data_ptr<float>();
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(ptr[i], 7.0f, 1e-5f);
    }
}

TEST(EdgeCaseTest, Numel) {
    EXPECT_EQ((Tensor<3, 4>::numel()), 12u);
    EXPECT_EQ((Tensor<2, 3, 5>::numel()), 30u);
    EXPECT_EQ((Tensor<1>::numel()), 1u);
    // Note: Tensor<>::numel() doesn't compile (empty fold expression) — by design
}

TEST(EdgeCaseTest, ScalarAlias) {
    // Scalar is Tensor<>
    Scalar s(torch::tensor(3.14f));
    ASSERT_EQ(s.dim(), 0);
    EXPECT_NEAR(s.item<float>(), 3.14f, 1e-4f);
}

TEST(EdgeCaseTest, ReverseScalarOps) {
    auto t = Tensor<4>(torch::ones({4}) * 2.0f);
    // float + Tensor
    auto add = 10.0f + t;
    EXPECT_NEAR(add.t().sum().item<float>(), 48.0f, 1e-5f);  // 4*12
    // float - Tensor
    auto sub = 10.0f - t;
    EXPECT_NEAR(sub.t().sum().item<float>(), 32.0f, 1e-5f);  // 4*8
    // float * Tensor
    auto mul = 3.0f * t;
    EXPECT_NEAR(mul.t().sum().item<float>(), 24.0f, 1e-5f);  // 4*6
    // float / Tensor
    auto div = 10.0f / t;
    EXPECT_NEAR(div.t().sum().item<float>(), 20.0f, 1e-5f);  // 4*5
}


// ============================================================
// Edge cases: BatchTensor boundary conditions
// ============================================================

TEST(EdgeCaseTest, BatchTensorSingleSample) {
    // batch_size=1 is a degenerate but valid case
    auto raw = torch::randn({1, 8});
    BatchTensor<8> bt(raw);
    ASSERT_EQ(bt.batch_size(), 1);
    // All ops should still work
    auto out = bt.relu();
    ASSERT_EQ(out.batch_size(), 1);
    auto mean = bt.batch_mean();
    ASSERT_EQ(mean.dim(), 1);
    ASSERT_EQ(mean.t().size(0), 8);
    // bind<1> should succeed
    auto bound = bt.bind<1>();
    ASSERT_EQ(bound.t().size(0), 1);
}

TEST(EdgeCaseTest, BatchTensorDimMismatch) {
    // 1D tensor should fail for BatchTensor<8> (needs 2D: [B, 8])
    auto raw1d = torch::randn({8});
    EXPECT_THROW((BatchTensor<8>(raw1d)), std::runtime_error);
    // 3D tensor should fail for BatchTensor<8> (needs 2D: [B, 8])
    auto raw3d = torch::randn({2, 8, 3});
    EXPECT_THROW((BatchTensor<8>(raw3d)), std::runtime_error);
}

TEST(EdgeCaseTest, BatchTensorHighDim) {
    // BatchTensor with 3 mathematical dims
    auto raw = torch::randn({4, 2, 3, 5});
    BatchTensor<2, 3, 5> bt(raw);
    ASSERT_EQ(bt.batch_size(), 4);
    ASSERT_EQ(bt.t().dim(), 4);
    // Activations
    auto r = bt.relu();
    ASSERT_EQ(r.batch_size(), 4);
    ASSERT_EQ(r.t().size(1), 2);
    ASSERT_EQ(r.t().size(2), 3);
    ASSERT_EQ(r.t().size(3), 5);
    // batch_mean collapses to Tensor<2,3,5>
    auto m = bt.batch_mean();
    ASSERT_EQ(m.dim(), 3);
    ASSERT_EQ(m.t().size(0), 2);
    ASSERT_EQ(m.t().size(1), 3);
    ASSERT_EQ(m.t().size(2), 5);
}

TEST(EdgeCaseTest, BatchTensorLargeBatch) {
    // Stress test with large batch — shape checks still work
    auto raw = torch::randn({1000, 4});
    BatchTensor<4> bt(raw);
    ASSERT_EQ(bt.batch_size(), 1000);
    auto out = bt.sigmoid();
    ASSERT_EQ(out.batch_size(), 1000);
    EXPECT_TRUE((out.t() > 0).all().item<bool>());
    EXPECT_TRUE((out.t() < 1).all().item<bool>());
}

// ============================================================
// Edge cases: nn pipe operator and Sequential
// ============================================================

TEST(EdgeCaseTest, PipeOperator) {
    // x | module should be equivalent to module.forward(x)
    trails::nn::Linear<4, 8, 16> linear;
    auto x = Tensor<4, 8>::randn();
    auto y_pipe = x | linear;
    auto y_fwd  = linear.forward(x);
    // Both should have same shape
    ASSERT_EQ(y_pipe.t().size(0), 4);
    ASSERT_EQ(y_pipe.t().size(1), 16);
    // Same values (same weights, same input)
    EXPECT_TRUE(torch::equal(y_pipe.t(), y_fwd.t()));
}

TEST(EdgeCaseTest, TensorOperatorWithRawTensor) {
    // Tensor + torch::Tensor (raw) should work
    auto t = Tensor<4>(torch::ones({4}) * 3.0f);
    auto raw = torch::ones({4}) * 2.0f;
    auto add = t + raw;
    EXPECT_NEAR(add.t().sum().item<float>(), 20.0f, 1e-5f);  // 4*5
    auto sub = t - raw;
    EXPECT_NEAR(sub.t().sum().item<float>(), 4.0f, 1e-5f);   // 4*1
    auto mul = t * raw;
    EXPECT_NEAR(mul.t().sum().item<float>(), 24.0f, 1e-5f);  // 4*6
    auto div = t / raw;
    EXPECT_NEAR(div.t().sum().item<float>(), 6.0f, 1e-5f);   // 4*1.5
}

TEST(EdgeCaseTest, TensorScalarOps) {
    // Tensor + float, Tensor - float
    auto t = Tensor<4>(torch::ones({4}) * 5.0f);
    auto add = t + 3.0f;
    EXPECT_NEAR(add.t().sum().item<float>(), 32.0f, 1e-5f);  // 4*8
    auto sub = t - 2.0f;
    EXPECT_NEAR(sub.t().sum().item<float>(), 12.0f, 1e-5f);  // 4*3
}