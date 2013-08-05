#include "smem_ops.h"
#include "equations.h"

namespace inplace {
namespace detail {

template __global__ void smem_row_shuffle(int m, int n, float* d, c2r::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, double* d, c2r::shuffle s);

template __global__ void smem_row_shuffle(int m, int n, int* d, c2r::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, long long* d, c2r::shuffle s);



}
}
