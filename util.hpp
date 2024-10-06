#pragma once

#include <torch/torch.h>
#include <cstdio>

namespace trainium {

// Dimension debugging
static inline void dim_debug_print(const torch::Tensor &t, const std::string &name) {
#ifndef NDEBUG
    printf("%s: dims=%zu, sizes=", name.c_str(), t.dim());
    for (int i = 0; i < t.dim(); i++)
    {
        printf("%zu ", t.size(i));
    }
    printf("\n");
#endif
}

#ifndef NDEBUG
#define dd(t) dim_debug_print(t, #t)
#else
#define dd(t, name) do {} while(0)
#endif

}