#include "smem_ops.h"

namespace inplace {
namespace detail {

template __global__ void smem_row_shuffle(int m, int n, float* d, shuffle s);
template __global__ void smem_row_shuffle(int m, int n, double* d, shuffle s);

template __global__ void smem_row_shuffle(int m, int n, int* d, shuffle s);
template __global__ void smem_row_shuffle(int m, int n, long long* d, shuffle s);



}
}
