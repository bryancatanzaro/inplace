#pragma once

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_col_op(int m, int n, T* d, T* tmp, F fn) {
    column_major_index cm(m, n);
    for(int j = blockIdx.x; j < n; j += gridDim.x) {
        fn.set_j(j); 
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            int src_idx = fn(i);
            tmp[cm(i, blockIdx.x)] = d[cm(src_idx, j)];
        }
        __syncthreads();
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            d[cm(i, j)] = tmp[cm(i, blockIdx.x)];
        }
        __syncthreads();
    }
}

template<typename T>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        column_major_index cm(m, n);
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[cm(i, s(j))];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[cm(i, j)] = tmp[rm(blockIdx.x, j)];
        }
        __syncthreads();
    }        
}

}
}
