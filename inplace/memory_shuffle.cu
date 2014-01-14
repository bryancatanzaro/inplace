#include "memory_ops.h"
#include "equations.h"

namespace inplace {
namespace detail {

//Work around nvcc/clang bug on OS X
#ifndef __clang__

template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, c2r::shuffle s);

template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, c2r::shuffle s);


template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, r2c::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, r2c::shuffle s);

template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, r2c::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, r2c::shuffle s);

#else
namespace {

template<typename A, typename B>
void* magic() {
    return (void*)&memory_row_shuffle<A, B>;
}


template void* magic<float, c2r::shuffle>();
template void* magic<double, c2r::shuffle>();
template void* magic<int, c2r::shuffle>();
template void* magic<long long, c2r::shuffle>();

template void* magic<float, r2c::shuffle>();
template void* magic<double, r2c::shuffle>();
template void* magic<int, r2c::shuffle>();
template void* magic<long long, r2c::shuffle>();

}

#endif

}
}
