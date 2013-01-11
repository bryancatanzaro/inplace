#pragma once

#include "temporary.h"

namespace inplace {

template<typename T>
__global__ void scatter_permute(int m, int n, int c, T* data, T* tmp) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            int dest_i = (i * n - (j * c)/m) % m;
        //     tmp[m * blockIdx.x + dest_i] = data[i + j * m];
            tmp[m * blockIdx.x + j] = dest_i;
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
             data[i + j * m] = tmp[m * blockIdx.x + j];
        }
        __syncthreads();
    }
}



template<typename T>
void transpose_cm(int m, int n, T* data, T* tmp_in=0) {
    temporary_storage<T> tmp(m, n, tmp_in);
    int c, k;
    extended_gcd(m, n, c, k);
    scatter_permute<<<n_ctas(), 1024>>>(m, n, c, data, static_cast<T*>(tmp));
}


}
