/* MNIST FC vs ConvNet comparison demo.
 * Trains two models on MNIST and compares their accuracy,
 * using Trails typed tensors for compile-time shape checking.
 */
#include <iostream>
#include <cstdio>
#include <torch/torch.h>
#include "trails.hpp"
#include "trails_nn.hpp"

using namespace trails;
namespace F = trails::functional;
namespace nn = trails::nn;

// ── MNIST constants ─────────────────────────────────────────
constexpr int kBatchSize = 64;
constexpr int kEpochs    = 5;

// Image dimensions
constexpr int kImgC      = 1;                              // input channels (grayscale)
constexpr int kImgH      = 28;                             // image height
constexpr int kImgW      = 28;                             // image width
constexpr int kImgPixels = kImgC * kImgH * kImgW;         // 784, flattened

// FC model
constexpr int kFCHidden1 = 256;
constexpr int kFCHidden2 = 128;

// ConvNet model
constexpr int kConv1Out  = 16;                             // conv1 output channels
constexpr int kConv2Out  = 32;                             // conv2 output channels
constexpr int kKernel    = 5;                              // conv kernel size
constexpr int kPool      = 2;                              // pool kernel & stride

// Derived dimensions (computed from above)
constexpr int kConv1H    = kImgH - kKernel + 1;           // 24
constexpr int kConv1W    = kImgW - kKernel + 1;           // 24
constexpr int kPool1H    = kConv1H / kPool;                // 12
constexpr int kPool1W    = kConv1W / kPool;                // 12
constexpr int kConv2H    = kPool1H - kKernel + 1;          // 8
constexpr int kConv2W    = kPool1W - kKernel + 1;          // 8
constexpr int kPool2H    = kConv2H / kPool;                // 4
constexpr int kPool2W    = kConv2W / kPool;                // 4
constexpr int kConvFlat  = kConv2Out * kPool2H * kPool2W;  // 512

constexpr int kNumClasses = 10;

// Training hyperparameters
constexpr double kLR        = 0.001;
constexpr double kMnistMean = 0.1307;
constexpr double kMnistStd  = 0.3081;

// ── FC-only model ──────────────────────────────────────────────
// Flatten(kImgH×kImgW→kImgPixels) → Linear(kImgPixels,kFCHidden1) → relu
// → Linear(kFCHidden1,kFCHidden2) → relu → Linear(kFCHidden2,kNumClasses)
struct FCModel : torch::nn::Module {
    std::shared_ptr<nn::Linear<kImgPixels, kFCHidden1>> fc1;
    std::shared_ptr<nn::Linear<kFCHidden1, kFCHidden2>> fc2;
    std::shared_ptr<nn::Linear<kFCHidden2, kNumClasses>> fc3;

    FCModel() {
        fc1 = register_module("fc1", std::make_shared<nn::Linear<kImgPixels, kFCHidden1>>());
        fc2 = register_module("fc2", std::make_shared<nn::Linear<kFCHidden1, kFCHidden2>>());
        fc3 = register_module("fc3", std::make_shared<nn::Linear<kFCHidden2, kNumClasses>>());
    }

    BatchTensor<kNumClasses> forward(BatchTensor<kImgC, kImgH, kImgW> input) {
        auto flat = F::flatten<0, 2>(input);             // BatchTensor<kImgPixels>
        auto h1 = fc1->forward(flat).relu();
        auto h2 = fc2->forward(h1).relu();
        return fc3->forward(h2);
    }
};

// ── ConvNet model ──────────────────────────────────────────────
// Conv2d(kImgC→kConv1Out, kKernel×kKernel) → relu → MaxPool2d(kPool,kPool)
// → Conv2d(kConv1Out→kConv2Out, kKernel×kKernel) → relu → MaxPool2d(kPool,kPool)
// → Flatten(kConv2Out×kPool2H×kPool2W=kConvFlat) → Linear(kConvFlat,kNumClasses)
struct ConvNetModel : torch::nn::Module {
    std::shared_ptr<nn::Conv2d<kImgC, kConv1Out, kKernel, kKernel>> conv1;
    std::shared_ptr<nn::Conv2d<kConv1Out, kConv2Out, kKernel, kKernel>> conv2;
    std::shared_ptr<nn::Linear<kConvFlat, kNumClasses>> fc;

    ConvNetModel() {
        conv1 = register_module("conv1",
            std::make_shared<nn::Conv2d<kImgC, kConv1Out, kKernel, kKernel>>());
        conv2 = register_module("conv2",
            std::make_shared<nn::Conv2d<kConv1Out, kConv2Out, kKernel, kKernel>>());
        fc = register_module("fc", std::make_shared<nn::Linear<kConvFlat, kNumClasses>>());
    }

    BatchTensor<kNumClasses> forward(BatchTensor<kImgC, kImgH, kImgW> input) {
        // Conv1 → relu → pool
        auto c1 = conv1->forward(input).relu();
        auto p1 = F::max_pool2d<kPool, kPool, kPool, kPool>(c1);   // → <B,kConv1Out,kPool1H,kPool1W>
        // Conv2 → relu → pool
        auto c2 = conv2->forward(p1).relu();
        auto p2 = F::max_pool2d<kPool, kPool, kPool, kPool>(c2);   // → <B,kConv2Out,kPool2H,kPool2W>
        // Flatten → FC
        auto flat = F::flatten<0, 2>(p2);                     // BatchTensor<kConvFlat>
        return fc->forward(flat);
    }
};

// ── Generic train + evaluate ───────────────────────────────────
template<typename Model>
void train_and_eval(Model& model, const std::string& data_path,
                    const std::string& name) {
    std::cout << "\n=== Training " << name << " ===\n";

    // Training data
    auto train_dataset = torch::data::datasets::MNIST(data_path)
        .map(torch::data::transforms::Normalize<>(kMnistMean, kMnistStd))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize));

    torch::optim::Adam optimizer(model.parameters(), /*lr=*/kLR);

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        model.train();
        double total_loss = 0;
        int batches = 0;
        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            auto input  = BatchTensor<kImgC, kImgH, kImgW>(batch.data);
            auto output = model.forward(input);
            auto loss   = F::cross_entropy(output, batch.target);
            loss.backward();
            optimizer.step();
            total_loss += loss.template item<double>();
            ++batches;
        }
        std::printf("Epoch %d/%d: avg loss = %.4f\n",
                    epoch, kEpochs, total_loss / batches);
    }

    // Test evaluation
    model.eval();
    auto test_dataset = torch::data::datasets::MNIST(
            data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(kMnistMean, kMnistStd))
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize));

    int correct = 0, total = 0;
    {
        torch::NoGradGuard no_grad;
        for (auto& batch : *test_loader) {
            auto input  = BatchTensor<kImgC, kImgH, kImgW>(batch.data);
            auto output = model.forward(input);
            auto pred   = output.t().argmax(1);
            correct += pred.eq(batch.target).sum().template item<int>();
            total   += batch.data.size(0);
        }
    }
    std::printf("%s Test Accuracy: %.2f%%\n",
                name.c_str(), 100.0 * correct / total);
}

// ── main ───────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string data_path = argc > 1 ? argv[1] : "./data";

    FCModel fc_model;
    train_and_eval(fc_model, data_path, "FC Model");

    ConvNetModel conv_model;
    train_and_eval(conv_model, data_path, "ConvNet Model");

    return 0;
}

