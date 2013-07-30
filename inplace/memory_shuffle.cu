#include "memory_ops.h"

namespace inplace {
namespace detail {

template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, shuffle s);

template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, shuffle s);



}
}
