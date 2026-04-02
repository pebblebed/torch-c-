// Copyright 2025, Pebblebed Management, LLC. All rights reserved.

#pragma once

#include "trails/trails.hpp"

namespace trainium {

template<int B,
    int Hidden, int EmbeddingDim, int DictionarySize,
    class HiddenStateEncoder,
    class Decoder>
class RNN {
    trails::Tensor<DictionarySize, EmbeddingDim> embed;
    HiddenStateEncoder enc;
    Decoder dec;

public:
    using input_t = trails::Tensor<B, 1>;
    using hidden_t = trails::Tensor<B, Hidden>;
    using output_t = trails::Tensor<B, DictionarySize>;

    RNN()
    : embed(torch::randn({DictionarySize, EmbeddingDim}))
    , enc()
    , dec() {}

    output_t forward(input_t input, hidden_t &hidden) {
        auto embedded = torch::nn::functional::embedding(input.t(), embed.t());
        auto projection = trails::Tensor<B, EmbeddingDim>(embedded.reshape({B, EmbeddingDim}));
        hidden = enc.forward(projection);
        return dec.forward(hidden);
    }
};

} // namespace trainium
