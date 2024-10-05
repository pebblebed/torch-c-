#include <iostream>
#include "charformer.hpp"
#include "dataset_dir.hpp"

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
    std::cout << net.forward("hello") << "\n";
    return 0;
}
