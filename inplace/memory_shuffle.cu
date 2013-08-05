#include "memory_ops.h"
#include "equations.h"

namespace inplace {
namespace detail {

template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, c2r::shuffle s);

template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, c2r::shuffle s);



}
}
