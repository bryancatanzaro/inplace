#pragma once

#include "introspect.h"

namespace inplace {

struct prerotator {
    int m_m, m_n, m_c;
    __host__ __device__
    prerotator(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__
    int operator()(const int& j) {
        return (j * m_c)/m_n;
    }
};

struct postpermuter {
    int m_m, m_n, m_c, m_j;
    __host__ __device__
    postpermuter(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__
    void set_j(const int& j) {
        m_j = j;
    }
    __host__ __device__
    int operator()(const int& i) {
        return ((i*m_n)-(i*m_c)/m_m+m_j) % m_m;
    }
};

template<typename T, typename F>
__global__ void col_rotate(int m, int n, T* d, T* tmp, F fn) {
    column_major_index cm(m, n);
    for(int j = blockIdx.x; j < n; j += gridDim.x) {
        int rotation_amount = fn(j);
        if (rotation_amount > 0) {            
            //if (rotation_amount < m/2) {
            for(int i = threadIdx.x; i < rotation_amount; i += blockDim.x) {
                tmp[cm(i, blockIdx.x)] = d[cm(i, j)];
            }
            __syncthreads();
            int n_blocks = (m - rotation_amount - 1)/(blockDim.x) + 1;
            int index = threadIdx.x + rotation_amount;
            for(int i = 0; i < n_blocks; i++) {
                T tmp;
                if (index < m) {
                    tmp = d[cm(index, j)];
                }
                __syncthreads();
                if (index < m) {
                    d[cm(index-rotation_amount, j)] = tmp;
                }
                index += blockDim.x;
            }
            __syncthreads();
            for(int i = threadIdx.x; i < rotation_amount; i += blockDim.x) {
                d[cm(i+m-rotation_amount, j)] = tmp[cm(i, blockIdx.x)];
            }
            __syncthreads();
            // } else {

            //     for(int i = threadIdx.x; i < m - rotation_amount; i += blockDim.x) {
            //         tmp[cm(i, blockIdx.x)] = d[cm(i + rotation_amount, j)];
            //     }
            //     __syncthreads();
            //     int n_blocks = (rotation_amount - 1)/(blockDim.x) + 1;
            //     int index = threadIdx.x;
            //     for(int i = 0; i < n_blocks; i++) {
            //         T tmp;
            //         if (index < rotation_amount) {
            //             tmp = d[cm(index, j)];
            //         }
            //         __syncthreads();
            //         if (index < rotation_amount) {
            //             d[cm(index+m-rotation_amount, j)] = tmp;
            //         }
            //         index += blockDim.x;
            //     }
            //     __syncthreads();
            //     for(int i = threadIdx.x; i < m - rotation_amount; i += blockDim.x) {
            //         d[cm(i, j)] = tmp[cm(i, blockIdx.x)];
            //     }
            //     __syncthreads();
            // }
        }
    }
}

template<typename T, typename F>
__global__ void col_op(int m, int n, T* d, T* tmp, F fn) {
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

struct shuffle {
    int m_m, m_n, m_c, m_k;
    __host__ __device__
    shuffle(int m, int n, int c, int k) : m_m(m), m_n(n), m_c(c), m_k(k) {}

    //This returns long long to avoid integer overflow in intermediate
    //computation
    __host__ __device__
    long long f(const int& i, const int& j) {
        long long r = j + i *(m_n - 1);
        if (i < (m_m + 1 - m_c + (j % m_c))) {
            return r;
        } else {
            return r + m_m;
        }
    }
    
    __host__ __device__
    int operator()(const int& i, const int& j) {
        long long fij = f(i, j);
        int term1 = (m_k *(fij/m_c)) % (m_n/m_c);
        int term2 = (fij % m_c) * (m_n/m_c);
        return (term1 + term2) % m_n;
    }
};

template<typename T>
__global__ void row_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        column_major_index cm(m, n);
        row_major_index rm(m, n);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[cm(i, s(i, j))];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[cm(i, j)] = tmp[rm(blockIdx.x, j)];
        }
        __syncthreads();
    }        
}

template<typename T>
void transpose(bool row_major, int m, int n, T* data, T* tmp_in=0) {
    if (!row_major) {
        std::swap(m, n);
    }

    temporary_storage<T> tmp(m, n, tmp_in);
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

    int blockdim = n_ctas();
    int threaddim = n_threads();

    if (c > 1) {
        col_rotate<<<blockdim, threaddim>>>
            (m, n, data, static_cast<T*>(tmp),
             prerotator(m, n, c));
    }
    row_shuffle<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp), shuffle(m, n, c, k));
    col_op<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp),
         postpermuter(m, n, c));
}

}
