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

namespace inplace {
namespace detail {

template __global__ void register_row_shuffle<SM, INSTANTIATED_TYPE, c2r::shuffle, WPT>(int, int, INSTANTIATED_TYPE*, c2r::shuffle);

}
}
