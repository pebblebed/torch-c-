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

constexpr int kBatchSize = 64;
constexpr int kEpochs = 5;

// ── FC-only model ──────────────────────────────────────────────
// Flatten(28×28→784) → Linear(784,256) → relu → Linear(256,128) → relu → Linear(128,10)
struct FCModel : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    FCModel() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, 10));
    }

    Tensor<kBatchSize, 10> forward(Tensor<kBatchSize, 1, 28, 28> input) {
        auto flat = F::flatten<1, 3>(input);            // Tensor<64, 784>
        auto h1 = Tensor<kBatchSize, 256>(fc1->forward(flat.t()));
        auto h2 = Tensor<kBatchSize, 128>(fc2->forward(F::relu(h1).t()));
        return Tensor<kBatchSize, 10>(fc3->forward(F::relu(h2).t()));
    }
};

// ── ConvNet model ──────────────────────────────────────────────
// Conv2d(1→16, 5×5) → relu → MaxPool2d(2,2)
// → Conv2d(16→32, 5×5) → relu → MaxPool2d(2,2)
// → Flatten(32×4×4=512) → Linear(512,10)
struct ConvNetModel : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc{nullptr};

    ConvNetModel() {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5)));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5)));
        fc = register_module("fc", torch::nn::Linear(512, 10));
    }

    Tensor<kBatchSize, 10> forward(Tensor<kBatchSize, 1, 28, 28> input) {
        // Conv1 → relu → pool
        auto c1 = Tensor<kBatchSize, 16, 24, 24>(conv1->forward(input.t()));
        auto p1 = F::max_pool2d<2, 2, 2, 2>(F::relu(c1));   // Tensor<64,16,12,12>
        // Conv2 → relu → pool
        auto c2 = Tensor<kBatchSize, 32, 8, 8>(conv2->forward(p1.t()));
        auto p2 = F::max_pool2d<2, 2, 2, 2>(F::relu(c2));   // Tensor<64,32,4,4>
        // Flatten → FC
        auto flat = F::flatten<1, 3>(p2);                     // Tensor<64, 512>
        return Tensor<kBatchSize, 10>(fc->forward(flat.t()));
    }
};

// ── Generic train + evaluate ───────────────────────────────────
template<typename Model>
void train_and_eval(Model& model, const std::string& data_path,
                    const std::string& name) {
    std::cout << "\n=== Training " << name << " ===\n";

    // Training data
    auto train_dataset = torch::data::datasets::MNIST(data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).drop_last(true));

    torch::optim::Adam optimizer(model.parameters(), /*lr=*/0.001);

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        model.train();
        double total_loss = 0;
        int batches = 0;
        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            auto input  = Tensor<kBatchSize, 1, 28, 28>(batch.data);
            auto output = model.forward(input);
            auto loss   = torch::nn::functional::cross_entropy(output.t(),
                                                               batch.target);
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
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).drop_last(true));

    int correct = 0, total = 0;
    {
        torch::NoGradGuard no_grad;
        for (auto& batch : *test_loader) {
            auto input  = Tensor<kBatchSize, 1, 28, 28>(batch.data);
            auto output = model.forward(input);
            auto pred   = output.t().argmax(1);
            correct += pred.eq(batch.target).sum().template item<int>();
            total   += kBatchSize;
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

