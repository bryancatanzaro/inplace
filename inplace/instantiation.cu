#include "equations.h"
#include "register_ops.h"
#include "sm.h"

#ifndef INSTANTIATED_TYPE
#define INSTANTIATED_TYPE double
#endif

#ifndef WPT
#define WPT 5
#endif

#ifndef SM
#define SM sm_20
#endif

#ifndef DIRECTION
#define DIRECTION c2r
#endif

namespace inplace {
namespace detail {

//Work around nvcc/clang bug on OS X
#ifndef __clang__

template __global__ void register_row_shuffle<SM, INSTANTIATED_TYPE, DIRECTION::shuffle, WPT>(int, int, INSTANTIATED_TYPE*, DIRECTION::shuffle);

#else
namespace {

template<typename A, typename B, typename C, int D>
void* magic() {
    return (void*)&register_row_shuffle<A, B, C, D>;
}

template void* magic<SM, INSTANTIATED_TYPE, DIRECTION::shuffle, WPT>();

}

#endif
          

}
}
