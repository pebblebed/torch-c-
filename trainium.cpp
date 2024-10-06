#include <iostream>
#include "charformer.hpp"
#include "dataset_dir.hpp"
#include "util.hpp"

using namespace trainium;

const int B = 12;
const int D = 64;
const int L = 1024;
const int H = 12;
const int layers = 17;

int main(int argc, char** argv) {
    auto t = torch::rand({2, 3}).cuda();
    auto sq = (t * t).mean();
    std::cout << t << "\n";
    std::cout << sq << "\n";
    std::cout << t / sq << "\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " dataset_dir\n";
        return 1;
    }

    std::string dataset_dir = argv[1];
    auto dataset = DatasetDir(dataset_dir, B, L);
    auto net = CharFormer<256, 128, 8, 12>();
    net.to(torch::kCUDA);

    auto train_test_val = train_test_val_split(dataset, 0.8, 0.1);
    auto &train = std::get<0>(train_test_val);
    auto &test = std::get<1>(train_test_val);
    auto &val = std::get<2>(train_test_val);

    auto dataloader = torch::data::make_data_loader(train.map(torch::data::transforms::Stack<>()),
        torch::data::samplers::SequentialSampler(train.size().value()),
        torch::data::DataLoaderOptions().batch_size(B));
    torch::optim::AdamW optimizer(net.parameters(), 0.1e-3);
    size_t i = 0;
    for (auto& batch : *dataloader) {
        batch.data.to(torch::kCUDA);
        batch.target.to(torch::kCUDA);
        dd(batch.data);
        dd(batch.target);
        printf("batch.data.dims(): %zu\n", batch.data.dim());
        for (int i = 0; i < batch.data.dim(); i++) {
            printf("batch.data.size(%d): %zu\n", i, batch.data.size(i));
        }
        optimizer.zero_grad();
        auto probs = net.forward(batch.data);
        auto loss = torch::nll_loss(torch::log(probs), batch.target);
        loss.backward();
        optimizer.step();
        if (i++ % 100 == 0) {
            std::cout << loss.item<float>() << "\n";
        }
    }
    return 0;
}
