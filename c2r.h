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

struct postrotate {
    int m_m;
    __host__ __device__
    postrotate(int m) : m_m(m) {}
    __host__ __device__
    int operator()(const int& j) {
        return j % m_m;
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

struct permuter {
    int m_m, m_n, m_c;
    __host__ __device__
    permuter(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__
    void set_j(const int& j) {}
    __host__ __device__
    int operator()(const int& i) {
        return (i * m_n - (i*m_c)/m_m) % m_m;
    }
};

struct identity {
    __host__ __device__
    void set_j(const int& j){}
    __host__ __device__
    int operator()(const int& i) {
        return i;
    }
};


template<typename T, typename F0, typename F1>
__global__ void rm_col_op(int m, int n, T* d, T* tmp, F0 fn0, F1 fn1=identity()) {
    column_major_index cm(m, n);
    for(int j = blockIdx.x; j < n; j += gridDim.x) {
        fn0.set_j(j); fn1.set_j(j);
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            int src_idx = fn0(i);
            tmp[cm(i, blockIdx.x)] = d[cm(src_idx, j)];
            //tmp[blockIdx.x * m + i] = d[j + src_idx * n];
        }
        __syncthreads();
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            int src_idx = fn1(i);
            d[cm(i, j)] = tmp[cm(src_idx, blockIdx.x)];
            //d[j + i * n] = tmp[blockIdx.x * m + src_idx];
        }
        __syncthreads();
    }
}

struct shuffle {
    int m_m, m_n, m_c, m_k;
    __host__ __device__
    shuffle(int m, int n, int c, int k) : m_m(m), m_n(n), m_c(c), m_k(k) {}

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
__global__ void rm_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        column_major_index cm(m, n);
        row_major_index rm(m, n);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[cm(i, s(i, j))];
            //tmp[blockIdx.x * n + j] = d[i * n + s(i, j)];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[cm(i, j)] = tmp[rm(blockIdx.x, j)];
            //d[i * n + j] = tmp[blockIdx.x * n + j];
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
    extended_gcd(m/c, n/c, t, k);

    int blockdim = n_ctas();
    int threaddim = n_threads();

    rm_col_op<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp), rotator<prerotate>(prerotate(m, n, c)), identity());
    rm_shuffle<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp), shuffle(m, n, c, k));
    rm_col_op<<<blockdim, threaddim>>>
        (m, n, data, static_cast<T*>(tmp), rotator<postrotate>(postrotate(m)), permuter(m, n, c));
}

}
