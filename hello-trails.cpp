#include <iostream>
#include <torch/torch.h>
#include "trails.hpp"

using namespace trails;

template<int A, int B, int C>
Tensor<A, C>
scaleAndMul(float k) {
    auto M1 = Tensor<A, B>::randn();
    auto M2 = Tensor<B, C>::randn();
    std::cout << "M1 (" << A << "x" << B << "):\n" << M1 << "\n\n";
    std::cout << "M2 (" << B << "x" << C << "):\n" << M2 << "\n\n";
    auto scaled = M1 * k;
    std::cout << "M1 * " << k << ":\n" << scaled << "\n\n";
    auto result = matmul(scaled, M2);
    std::cout << "Result (" << A << "x" << C << "):\n" << result << "\n";
    return result;
}

int main() {
    std::cout << "=== Hello Trails! ===\n\n";
    auto r = scaleAndMul<3, 4, 2>(2.5f);
    // The return type is Tensor<3, 2> — enforced at compile time!
    return 0;
}

