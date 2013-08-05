#pragma once
#include "index.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, F s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[rm(i, s(j))];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = tmp[rm(blockIdx.x, j)];
        }
        __syncthreads();
    }        
}

}
}
