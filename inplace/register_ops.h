#pragma once
#include "array.h"
#include "index.h"

namespace inplace {
namespace detail {

template<typename T, int R, typename F>
struct gather_row_impl {
    static __device__ __forceinline__ void fun(int i, int j, row_major_index rm, const T* d, array<T, R>& s, F& fn) {
        if (j < rm.n) {
            s.head = d[rm(i, fn(j))];
            gather_row_impl<T, R-1, F>::fun(i, j + blockDim.x, rm, d, s.tail, fn);

        }
    }
};

template<typename T, typename F>
struct gather_row_impl<T, 1, F> {
    static __device__ __forceinline__ void fun(int i, int j, row_major_index rm, const T* d, array<T, 1>& s, F& fn) {
        if (j < rm.n) {
            s.head = d[rm(i, fn(j))];
        }
    }
};  

template<typename T, int R, typename F>
__device__ __forceinline__ void gather_row(int i, row_major_index rm,
                           const T* d, array<T, R>& s, F& fn) {
    gather_row_impl<T, R, F>::fun(i, threadIdx.x, rm, d, s, fn);
}

template<typename T, int R>
struct write_row_impl {
    static __device__ __forceinline__ void fun(const int& i, const int& j,
                               const row_major_index& rm,
                               const array<T, R>& s, T* d) {
        if (j < rm.n) {
            d[rm(i, j)] = s.head;
            write_row_impl<T, R-1>::fun(i, j + blockDim.x, rm, s.tail, d);
        }
    }
};

template<typename T>
struct write_row_impl<T, 1> {
    static __device__ __forceinline__ void fun(const int& i, const int& j,
                               const row_major_index& rm,
                               const array<T, 1>& s, T* d) {
        if (j < rm.n) {
            d[rm(i, j)] = s.head;
        }
    }
};  


template<typename T, int R>
__device__ __forceinline__ void write_row(const int& i, const row_major_index& rm, 
                          const array<T, R>& s, T* d) {
    write_row_impl<T, R>::fun(i, threadIdx.x, rm, s, d);
}



template<typename SM, typename T, int R>
__global__ void register_row_shuffle(int m, int n, T* d, shuffle s) {
    row_major_index rm(m, n);
    array<T, R> thread_storage;

    int i = blockIdx.x;
    s.set_i(i);
    
    gather_row(i, rm, d, thread_storage, s);

    __syncthreads();

    write_row(i, rm, thread_storage, d);

}

}
}
