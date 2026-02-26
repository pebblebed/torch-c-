#pragma once

#include <torch/torch.h>
#include <cstdio>

namespace trainium {

// Dimension debugging
static inline void dim_debug_print(const torch::Tensor &t, const std::string &name) {
#ifndef NDEBUG
    printf("%s: dims=%lld, sizes=", name.c_str(), static_cast<long long>(t.dim()));
    for (int64_t i = 0; i < t.dim(); i++)
    {
        printf("%lld ", static_cast<long long>(t.size(i)));
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
