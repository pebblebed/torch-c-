#include <iostream>
#include "charformer.hpp"
#include "dataset_dir.hpp"
#include "util.hpp"
#include "trails/trails.hpp"

using namespace trainium;
using namespace trails;

// CharFormer hyperparameters (small enough for CPU training)
constexpr int B = 8;
constexpr int SeqLen = 128;
constexpr int VocabSize = 256;
constexpr int ModelDim = 128;
constexpr int NumHeads = 4;
constexpr int FFDim = 512;
constexpr int NLayers = 4;

// n_ctx for the dataset: we need SeqLen input tokens + 1 for the shifted target
constexpr int N_CTX = SeqLen + 1;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " dataset_dir\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::cerr << "Loading dataset from: " << dataset_path << std::endl;
    auto dataset = DatasetDir(dataset_path, B, N_CTX);

    auto net = CharFormer<SeqLen, VocabSize, ModelDim, NumHeads, FFDim, NLayers>();

    auto train_test_val = train_test_val_split(dataset, 0.1, 0.1);
    auto &train = std::get<0>(train_test_val);

    auto dataloader = torch::data::make_data_loader(
        train.map(torch::data::transforms::Stack<>()),
        torch::data::samplers::SequentialSampler(train.size().value()),
        torch::data::DataLoaderOptions().batch_size(B));

    torch::optim::AdamW optimizer(
        net.parameters(),
        torch::optim::AdamWOptions(1e-4));

    std::cerr << "Starting training..." << std::endl;
    size_t step = 0;
    for (auto& batch : *dataloader) {
        auto data = batch.data;    // (B, SeqLen)
        auto target = batch.target; // (B, SeqLen)

        // Skip incomplete batches
        if (data.size(0) != B) {
            continue;
        }

        optimizer.zero_grad();

        // Forward pass: input BatchTensor<SeqLen> -> logits BatchTensor<SeqLen, VocabSize>
        auto input = BatchTensor<SeqLen>(data);
        auto logits = net.forward(input);

        // Reshape for cross_entropy: flatten (batch, SeqLen, VocabSize) -> (batch*SeqLen, VocabSize)
        auto batch_size = logits.batch_size();
        auto loss = language_model_loss(logits.t(), target);
        loss.backward();
        optimizer.step();

        if (step % 10 == 0) {
            std::cout << "step " << step << " loss: " << loss.item<float>() << std::endl;
        }
        step++;
    }

    std::cerr << "Training complete. " << step << " steps." << std::endl;
    return 0;
}
