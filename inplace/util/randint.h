#pragma once
#include <cstdlib>

namespace inplace {
namespace detail {
__host__
inline int randint(int lb, int ub) {
    int span = ub - lb;
    unsigned int r = rand() % span;
    return r + lb;
}
}
}
