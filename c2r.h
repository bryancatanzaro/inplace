#pragma once

#include "introspect.h"

namespace inplace {

struct prerotate {
    int m_m, m_n, m_c;
    __host__ __device__
    prerotate(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__
    int operator()(const int& j) {
        return (j * m_c)/m_n;
    }
};

template<typename F>
struct rotator {
    F m_f;
    int m_a;
    __host__ __device__
    rotator(F f) : m_f(f) {}
    __host__ __device__
    void set_j(const int& j) {
        m_a = m_f(j);
    }
    __host__ __device__
    int operator()(const int& i) {
        return (i + m_a) % m_f.m_m;
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

    col_op<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp),
         rotator<prerotate>(prerotate(m, n, c)));
    row_shuffle<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp), shuffle(m, n, c, k));
    col_op<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp),
         postpermuter(m, n, c));
}

}
