#include <iostream>
#include "charformer.hpp"

using namespace trainium;

const int B = 12;
const int D = 64;
const int H = 12;

const int layers = 17;

int main(int argc, char** argv) {
    auto t = torch::rand({2, 3}).cuda();
    auto sq = (t * t).mean();
    std::cout << t << "\n";
    std::cout << sq << "\n";
    std::cout << t / sq << "\n";

    auto net = CharFormer<256, 128, 8, 12>();
    std::cout << net.forward("hello") << "\n";
    return 0;
}
