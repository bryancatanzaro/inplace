#pragma once

namespace inplace {
namespace detail {


//Work around extern shared aliasing problem
template<class T>
struct shared_memory{};

template<>
struct shared_memory<float> {
    __device__
    operator float*() const {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template<>
struct shared_memory<double> {
    __device__
    operator double*() const {
        extern __shared__ double s_double[];
        return s_double;
    }
};



template<typename T>
__global__ void smem_row_shuffle(int m, int n, T* d, shuffle s) {
    T* shared_row = static_cast<T*>(shared_memory<T>());
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            shared_row[j] = d[rm(i, j)];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = shared_row[s(j)];
        }
        __syncthreads();
    }        
}

}
}
